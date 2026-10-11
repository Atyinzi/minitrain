"""End-to-end Transformer benchmark for CS336 Assignment 2.

Implemented by Codex. Supports warmup, per-phase timing, BF16 autocast,
torch.compile, NVTX capture, and CUDA memory snapshots.
"""

from __future__ import annotations

import argparse
import json
import statistics
import timeit
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10b": (4608, 12288, 50, 36),
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device: torch.device) -> float:
    synchronize(device)
    start = timeit.default_timer()
    fn()
    synchronize(device)
    return (timeit.default_timer() - start) * 1_000


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--mode", choices=("forward", "backward", "train"), default="train")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--attention-backend", choices=("naive", "sdpa", "triton"), default="naive")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--memory-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; correctness tests may still run on CPU.")
    device = torch.device("cuda")
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attention_backend=args.attention_backend,
    ).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters())
    inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    def amp():
        if args.mixed_precision:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def one_step(record: bool) -> dict[str, float]:
        times: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        box = {}

        def forward():
            with torch.cuda.nvtx.range("forward"), amp():
                logits = model(inputs)
                box["loss"] = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        elapsed = timed_call(forward, device)
        if record:
            times["forward_ms"] = elapsed
        if args.mode in ("backward", "train"):
            elapsed = timed_call(lambda: box["loss"].backward(), device)
            if record:
                times["backward_ms"] = elapsed
        if args.mode == "train":
            elapsed = timed_call(optimizer.step, device)
            if record:
                times["optimizer_ms"] = elapsed
        return times

    for _ in range(args.warmup):
        one_step(record=False)

    # Warmup allocates optimizer state and compiles kernels. Start peak-memory
    # measurement only after that one-time work has completed.
    torch.cuda.reset_peak_memory_stats(device)

    if args.memory_snapshot:
        args.memory_snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    measurements = []
    with torch.cuda.nvtx.range("measurement"):
        for _ in range(args.steps):
            measurements.append(one_step(record=True))

    if args.memory_snapshot:
        torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)

    total_step_ms = [sum(row.values()) for row in measurements]
    mean_step_ms = statistics.mean(total_step_ms)
    tokens_per_step = args.batch_size * args.context_length
    result = {
        "configuration": vars(args) | {"device": torch.cuda.get_device_name(device)},
        "timings": {key: summarize([row[key] for row in measurements]) for key in measurements[0]},
        "total_step_ms": summarize(total_step_ms),
        "tokens_per_second": tokens_per_step / (mean_step_ms / 1_000),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

