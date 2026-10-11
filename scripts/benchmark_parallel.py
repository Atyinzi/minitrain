"""DDP, optimizer-sharding, and FSDP benchmark for MiniTrain.

Implemented by Codex. Launch with ``torchrun --standalone --nproc-per-node=2``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.distributed import DistributedDataParallel, FullyShardedDataParallel, ShardedOptimizer
from scripts.benchmark_transformer import MODEL_CONFIGS


def tensor_storage_bytes(value) -> int:
    """Count tensor payload bytes in a nested optimizer state dict."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_storage_bytes(item) for item in value)
    return 0


class PostBackwardDDP(nn.Module):
    """Baseline DDP supporting individual or one flattened all-reduce."""

    def __init__(self, module: nn.Module, flatten: bool):
        super().__init__()
        self.module = module
        self.flatten = flatten
        for tensor in module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def synchronize(self) -> float:
        torch.cuda.synchronize()
        start = timeit.default_timer()
        parameters = [p for p in self.module.parameters() if p.grad is not None]
        if self.flatten:
            flat = torch._utils._flatten_dense_tensors([p.grad for p in parameters])
            dist.all_reduce(flat)
            flat.div_(dist.get_world_size())
            for parameter, gradient in zip(parameters, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in parameters])):
                parameter.grad.copy_(gradient)
        else:
            for parameter in parameters:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(dist.get_world_size())
        torch.cuda.synchronize()
        return (timeit.default_timer() - start) * 1_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=("naive", "flat", "overlap", "sharded_optimizer", "fsdp_no_prefetch", "fsdp"),
        required=True,
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/parallel.json"))
    args = parser.parse_args()
    rank, local_rank, world_size = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[args.model_size]
    base = BasicsTransformerLM(10_000, args.context_length, d_model, num_layers, num_heads, d_ff).to(device)
    if args.strategy in ("fsdp", "fsdp_no_prefetch"):
        model = FullyShardedDataParallel(
            base,
            compute_dtype=torch.bfloat16,
            prefetch=args.strategy == "fsdp",
        )
    elif args.strategy in ("naive", "flat"):
        model = PostBackwardDDP(base, flatten=args.strategy == "flat")
    else:
        model = DistributedDataParallel(base)
    if args.strategy == "sharded_optimizer":
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    local_batch = args.batch_size // world_size
    tokens = torch.randint(10_000, (local_batch, args.context_length), device=device)
    targets = torch.randint(10_000, (local_batch, args.context_length), device=device)

    def step() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = timeit.default_timer()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        if isinstance(model, PostBackwardDDP):
            communication_ms = model.synchronize()
        else:
            communication_start = timeit.default_timer()
            model.finish_gradient_synchronization()
            torch.cuda.synchronize()
            communication_ms = (timeit.default_timer() - communication_start) * 1_000
        optimizer.step()
        torch.cuda.synchronize()
        return (timeit.default_timer() - start) * 1_000, communication_ms

    for _ in range(args.warmup):
        step()
    torch.cuda.reset_peak_memory_stats(device)
    samples = [step() for _ in range(args.steps)]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, samples)

    memory = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device) / 1024**2,
            torch.cuda.max_memory_reserved(device) / 1024**2,
        ],
        device=device,
    )
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    local_storage = torch.tensor(
        [
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024**2,
            tensor_storage_bytes(optimizer.state_dict()) / 1024**2,
        ],
        device=device,
    )
    storage_by_rank = [torch.empty_like(local_storage) for _ in range(world_size)]
    dist.all_gather(storage_by_rank, local_storage)
    if rank == 0:
        # A distributed step finishes when its slowest rank finishes. Taking
        # per-step maxima gives the latency users actually observe.
        iteration_samples = [max(rank_samples[index][0] for rank_samples in gathered) for index in range(args.steps)]
        communication_samples = [max(rank_samples[index][1] for rank_samples in gathered) for index in range(args.steps)]
        iteration_mean_ms = statistics.mean(iteration_samples)
        communication_mean_ms = statistics.mean(communication_samples)
        result = {
            "strategy": args.strategy,
            "configuration": {
                "model_size": args.model_size,
                "context_length": args.context_length,
                "global_batch_size": args.batch_size,
                "world_size": world_size,
                "warmup": args.warmup,
                "steps": args.steps,
                "device": torch.cuda.get_device_name(device),
            },
            "iteration_mean_ms": iteration_mean_ms,
            "iteration_std_ms": statistics.stdev(iteration_samples) if len(iteration_samples) > 1 else 0.0,
            "exposed_communication_mean_ms": communication_mean_ms,
            "exposed_communication_fraction": communication_mean_ms / iteration_mean_ms,
            "tokens_per_second": args.batch_size * args.context_length / (iteration_mean_ms / 1_000),
            "peak_allocated_mib": float(memory[0].cpu()),
            "peak_reserved_mib": float(memory[1].cpu()),
            "parameter_storage_mib_per_rank": [float(values[0].cpu()) for values in storage_by_rank],
            "optimizer_storage_mib_per_rank": [float(values[1].cpu()) for values in storage_by_rank],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
