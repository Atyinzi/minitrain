"""A small, readable training loop used as the Assignment 2 baseline."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_cosine_lr
from cs336_systems.distributed import DistributedDataParallel, FullyShardedDataParallel, ShardedOptimizer


MODEL_CONFIGS = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
}


def synchronize(device: torch.device) -> None:
    """Wait until CUDA has finished its queued work before measuring time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def optimizer_shard_path(checkpoint_path: Path, rank: int) -> Path:
    """Return the sidecar checkpoint path for one ZeRO-1 optimizer shard."""
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.optimizer_rank{rank}{checkpoint_path.suffix}")


def setup_distributed(
    requested_device: str,
    requested_backend: str | None,
) -> tuple[int, int, torch.device]:
    """Initialize torchrun when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if requested_device == "gpu":
            requested_device = "cuda"
        return 0, 1, torch.device(requested_device)

    local_rank = int(os.environ["LOCAL_RANK"])
    wants_cuda = requested_device == "gpu" or requested_device.startswith("cuda")
    if wants_cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = requested_backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = requested_backend or "gloo"

    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), device


def get_batch(
    data: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
    starts: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从连续 token 数据中随机取出输入和下一个 token 目标。"""
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError("数据长度必须大于 context_length")

    if starts is None:
        starts = np.random.randint(0, max_start + 1, size=batch_size)
    elif len(starts) != batch_size:
        raise ValueError("starts 的数量必须等于 batch_size")
    inputs = np.stack([data[s : s + context_length] for s in starts]).copy()
    targets = np.stack([data[s + 1 : s + context_length + 1] for s in starts]).copy()

    input_tensor = torch.from_numpy(inputs).long().to(device)
    target_tensor = torch.from_numpy(targets).long().to(device)
    return input_tensor, target_tensor


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_steps: int,
    mixed_precision: bool,
) -> float:
    """在固定且均匀分布的验证片段上计算平均 loss。"""
    model.eval()
    losses = []
    max_start = len(data) - context_length - 1
    validation_starts = np.linspace(
        0,
        max_start,
        num=eval_steps * batch_size,
        dtype=np.int64,
    )
    for eval_step in range(eval_steps):
        begin = eval_step * batch_size
        starts = validation_starts[begin : begin + batch_size]
        inputs, targets = get_batch(data, batch_size, context_length, device, starts=starts)
        if mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        else:
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        losses.append(float(loss.cpu()))
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Readable CS336 Assignment 2 training baseline")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-bin", type=Path, default=None)
    parser.add_argument("--val-bin", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--loss-ema-beta", type=float, default=0.98)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--attention-backend", choices=("naive", "sdpa", "triton"), default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default=None)
    parser.add_argument(
        "--parallel-strategy",
        choices=("ddp", "fsdp"),
        default="ddp",
        help="DDP replicates parameters; FSDP shards Linear/Embedding weights and gradients",
    )
    parser.add_argument(
        "--fsdp-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefetch the next FSDP parameter shard during current-layer compute",
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 keeps full AdamW state; 1 shards optimizer state across ranks",
    )
    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=float,
        default=25.0,
        help="Maximum DDP gradient bucket size in MiB",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/minitrain.pt"))
    parser.add_argument("--log-file", type=Path, default=Path("results/minitrain.jsonl"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps 必须大于等于 1")
    if args.eval_steps < 1:
        raise ValueError("--eval-steps 必须大于等于 1")
    if args.eval_every < 0:
        raise ValueError("--eval-every 必须大于等于 0")
    if args.ddp_bucket_cap_mb <= 0:
        raise ValueError("--ddp-bucket-cap-mb 必须大于 0")
    if args.parallel_strategy == "fsdp" and args.zero_stage == 1:
        raise ValueError("FSDP 已经分片优化器状态，不能同时使用 --zero-stage 1")
    if args.parallel_strategy == "fsdp" and args.compile:
        raise ValueError("当前教学版 FSDP 暂不支持同时使用 --compile")
    if not 0 <= args.loss_ema_beta < 1:
        raise ValueError("--loss-ema-beta 必须位于 [0, 1) 区间")
    rank, world_size, device = setup_distributed(args.device, args.distributed_backend)
    is_distributed = world_size > 1
    is_main_process = rank == 0
    if args.mixed_precision and device.type != "cuda":
        raise ValueError("--mixed-precision currently requires a CUDA device")

    # Different ranks sample different training sequences. The distributed
    # wrappers broadcast rank zero's initial parameters before training.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    train_data = None
    if args.train_bin is not None:
        train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
        if is_distributed:
            shard_start = len(train_data) * rank // world_size
            shard_end = len(train_data) * (rank + 1) // world_size
            train_data = train_data[shard_start:shard_end]
    val_data = None
    if args.val_bin is not None:
        val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")

    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[args.model_size]
    raw_model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        activation_checkpointing=args.activation_checkpointing,
        attention_backend=args.attention_backend,
    ).to(device)
    model: torch.nn.Module = raw_model
    if args.compile:
        model = torch.compile(model)
    if args.parallel_strategy == "fsdp":
        compute_dtype = torch.bfloat16 if args.mixed_precision else None
        fsdp_prefetch = args.fsdp_prefetch and not args.activation_checkpointing
        model = FullyShardedDataParallel(
            model,
            compute_dtype=compute_dtype,
            prefetch=fsdp_prefetch,
        )
        if is_main_process and args.fsdp_prefetch and args.activation_checkpointing:
            print("FSDP prefetch disabled because activation checkpointing recomputes layer forwards")
    elif is_distributed:
        model = DistributedDataParallel(model, bucket_cap_mb=args.ddp_bucket_cap_mb)

    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.zero_stage == 1:
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    start_step = 0
    previous_elapsed_seconds = 0.0
    loss_ema = None
    resume_training_args = {}
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_parallel_strategy = checkpoint.get("training_args", {}).get("parallel_strategy", "ddp")
        if saved_parallel_strategy != args.parallel_strategy:
            raise ValueError(
                f"checkpoint 使用 {saved_parallel_strategy}，当前并行策略是 {args.parallel_strategy}"
            )
        if args.parallel_strategy == "fsdp":
            model.load_full_state_dict(checkpoint["model_state_dict"])
        else:
            raw_model.load_state_dict(checkpoint["model_state_dict"])
        saved_zero_stage = checkpoint.get("training_args", {}).get("zero_stage", 0)
        if saved_zero_stage != args.zero_stage:
            raise ValueError(
                f"checkpoint 使用 ZeRO stage {saved_zero_stage}，当前参数是 stage {args.zero_stage}"
            )
        optimizer_state_is_sharded = checkpoint.get(
            "optimizer_state_sharded",
            "optimizer_shard_world_size" in checkpoint,
        )
        expects_sharded_optimizer_state = is_distributed and (
            args.zero_stage == 1 or args.parallel_strategy == "fsdp"
        )
        if optimizer_state_is_sharded != expects_sharded_optimizer_state:
            raise ValueError("checkpoint 的优化器分片方式与当前并行配置不一致")
        if expects_sharded_optimizer_state:
            saved_world_size = checkpoint.get("optimizer_shard_world_size")
            if saved_world_size != world_size:
                raise ValueError(
                    f"分片 checkpoint 使用 {saved_world_size} 个 rank，当前是 {world_size} 个"
                )
            shard_path = optimizer_shard_path(args.resume, rank)
            optimizer.load_state_dict(torch.load(shard_path, map_location=device, weights_only=False))
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        previous_elapsed_seconds = checkpoint.get("elapsed_time_seconds", 0.0)
        loss_ema = checkpoint.get("loss_ema")
        resume_training_args = checkpoint.get("training_args", {})
        rank_rng_states = checkpoint.get("rng_states_by_rank")
        if rank_rng_states is not None and len(rank_rng_states) == world_size:
            rank_rng_state = rank_rng_states[rank]
            torch.set_rng_state(rank_rng_state["torch"].cpu())
            np.random.set_state(rank_rng_state["numpy"])
            if device.type == "cuda" and rank_rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state(rank_rng_state["cuda"].cpu(), device=device)
        elif rank == 0:
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
                cuda_rng_states = [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
                torch.cuda.set_rng_state_all(cuda_rng_states)
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
        else:
            # Older single-process checkpoints contain only rank zero's RNG.
            torch.manual_seed(args.seed + rank + start_step)
            np.random.seed(args.seed + rank + start_step)
        if is_main_process:
            print(f"resumed from step {start_step}: {args.resume}")

    if args.cosine_lr:
        if args.lr_decay_steps is None:
            saved_decay_steps = resume_training_args.get("lr_decay_steps")
            args.lr_decay_steps = saved_decay_steps or (start_step + args.steps)
        if not 0 <= args.warmup_steps < args.lr_decay_steps:
            raise ValueError("学习率调度要求 0 <= warmup_steps < lr_decay_steps")
        if not 0 <= args.min_learning_rate <= args.learning_rate:
            raise ValueError("min_learning_rate 必须位于 0 和 learning_rate 之间")

    def autocast_context():
        if args.mixed_precision:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    if is_main_process:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()
    log_mode = "a" if args.resume is not None else "w"
    run_start = time.perf_counter()
    effective_batch_size = args.batch_size * args.grad_accum_steps * world_size
    tokens_per_step = effective_batch_size * args.context_length
    log_context = args.log_file.open(log_mode, encoding="utf-8") if is_main_process else nullcontext(None)
    with log_context as log_file:
        for local_step in range(1, args.steps + 1):
            step = start_step + local_step
            learning_rate = args.learning_rate
            if args.cosine_lr:
                learning_rate = get_cosine_lr(
                    step,
                    args.learning_rate,
                    args.min_learning_rate,
                    args.warmup_steps,
                    args.lr_decay_steps,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()

            loss_sum = torch.zeros((), device=device)
            for micro_step in range(args.grad_accum_steps):
                if train_data is None:
                    # 没有提供数据文件时，仍然可以使用随机数据测试性能。
                    inputs = torch.randint(
                        args.vocab_size,
                        (args.batch_size, args.context_length),
                        device=device,
                    )
                    targets = torch.randint(
                        args.vocab_size,
                        (args.batch_size, args.context_length),
                        device=device,
                    )
                else:
                    inputs, targets = get_batch(
                        train_data,
                        args.batch_size,
                        args.context_length,
                        device,
                    )

                sync_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel)
                    and micro_step < args.grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context():
                        logits = model(inputs)
                        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                    loss_sum += loss.detach()
                    (loss / args.grad_accum_steps).backward()

            if is_distributed:
                model.finish_gradient_synchronization()
            optimizer.step()
            synchronize(device)
            step_seconds = time.perf_counter() - start

            loss_metric = loss_sum / args.grad_accum_steps
            if is_distributed:
                dist.all_reduce(loss_metric, op=dist.ReduceOp.SUM)
                loss_metric.div_(world_size)
                step_time = torch.tensor(step_seconds, device=device)
                dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
                step_seconds = float(step_time.cpu())
            loss_value = float(loss_metric.cpu())
            if loss_ema is None:
                loss_ema = loss_value
            else:
                loss_ema = args.loss_ema_beta * loss_ema + (1 - args.loss_ema_beta) * loss_value

            peak_allocated_mib = None
            peak_reserved_mib = None
            if device.type == "cuda":
                memory_metrics = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device) / (1024**2),
                        torch.cuda.max_memory_reserved(device) / (1024**2),
                    ],
                    device=device,
                )
                if is_distributed:
                    dist.all_reduce(memory_metrics, op=dist.ReduceOp.MAX)
                peak_allocated_mib, peak_reserved_mib = [float(value) for value in memory_metrics.cpu()]

            should_evaluate = val_data is not None and args.eval_every > 0 and step % args.eval_every == 0
            if is_distributed and should_evaluate:
                dist.barrier()
            val_loss = None
            # FSDP forward contains collectives, so every rank must enter
            # evaluation. DDP can evaluate on rank zero alone.
            if should_evaluate and (is_main_process or args.parallel_strategy == "fsdp"):
                val_loss = evaluate(
                    model,
                    val_data,
                    args.batch_size,
                    args.context_length,
                    device,
                    args.eval_steps,
                    args.mixed_precision,
                )
            if is_distributed and should_evaluate:
                dist.barrier()

            if is_main_process:
                record = {
                    "step": step,
                    "loss": loss_value,
                    "loss_ema": loss_ema,
                    "learning_rate": learning_rate,
                    "attention_backend": args.attention_backend,
                    "parallel_strategy": args.parallel_strategy,
                    "zero_stage": args.zero_stage,
                    "world_size": world_size,
                    "step_ms": step_seconds * 1000,
                    "tokens_per_second": tokens_per_step / step_seconds,
                    "effective_batch_size": effective_batch_size,
                    "wall_time_seconds": previous_elapsed_seconds + (time.perf_counter() - run_start),
                }
                if peak_allocated_mib is not None:
                    record["peak_allocated_mib"] = peak_allocated_mib
                    record["peak_reserved_mib"] = peak_reserved_mib
                if val_loss is not None:
                    record["val_loss"] = val_loss
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                message = f"step {step:4d} | loss {record['loss']:.4f} | ema {record['loss_ema']:.4f}"
                if "val_loss" in record:
                    message += f" | val_loss {record['val_loss']:.4f}"
                message += f" | lr {record['learning_rate']:.2e}"
                message += f" | {record['tokens_per_second']:.0f} tok/s"
                if "peak_allocated_mib" in record:
                    message += f" | peak {record['peak_allocated_mib']:.0f} MiB"
                message += f" | time {record['step_ms']:.2f} ms"
                print(message, flush=True)

    elapsed_time_seconds = previous_elapsed_seconds + (time.perf_counter() - run_start)
    local_rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    if is_distributed:
        rng_states_by_rank = [None] * world_size if is_main_process else None
        dist.gather_object(local_rng_state, rng_states_by_rank, dst=0)
    else:
        rng_states_by_rank = [local_rng_state]

    optimizer_state_is_sharded = is_distributed and (
        args.zero_stage == 1 or args.parallel_strategy == "fsdp"
    )
    if optimizer_state_is_sharded:
        # Each rank writes only its local AdamW state. This avoids gathering all
        # shards onto rank zero and preserves the memory advantage.
        torch.save(optimizer.state_dict(), optimizer_shard_path(args.output, rank))
        dist.barrier()

    if args.parallel_strategy == "fsdp":
        checkpoint_model_state = model.gather_full_state_dict_rank0()
    else:
        checkpoint_model_state = raw_model.state_dict() if is_main_process else None

    if is_main_process:
        checkpoint = {
            "model_state_dict": checkpoint_model_state,
            "model_config": {
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "activation_checkpointing": args.activation_checkpointing,
                "attention_backend": args.attention_backend,
            },
            "training_args": dict(vars(args)),
            "step": start_step + args.steps,
            "elapsed_time_seconds": elapsed_time_seconds,
            "loss_ema": loss_ema,
            "torch_rng_state": local_rng_state["torch"],
            "numpy_rng_state": local_rng_state["numpy"],
            "rng_states_by_rank": rng_states_by_rank,
        }
        if optimizer_state_is_sharded:
            checkpoint["optimizer_shard_world_size"] = world_size
            checkpoint["optimizer_state_sharded"] = True
        else:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        checkpoint["training_args"]["output"] = str(args.output)
        checkpoint["training_args"]["log_file"] = str(args.log_file)
        torch.save(checkpoint, args.output)
        print(f"saved checkpoint to {args.output}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
