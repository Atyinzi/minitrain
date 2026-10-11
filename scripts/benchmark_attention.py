"""Attention sweep for CS336 Assignment 2.

Implemented by Codex. Writes machine-readable CSV and records OOMs rather than
aborting the entire sweep.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import timeit
from pathlib import Path

import torch

from cs336_systems.attention import FlashAttention2Triton


def attention(q, k, v, causal=True):
    scores = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
    if causal:
        n = q.shape[-2]
        scores.masked_fill_(torch.arange(n, device=q.device)[:, None] < torch.arange(n, device=q.device)[None, :], -1e6)
    return torch.softmax(scores, dim=-1) @ v


def bench(fn, repetitions: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = timeit.default_timer()
        fn()
        torch.cuda.synchronize()
        samples.append((timeit.default_timer() - start) * 1_000)
    return sum(samples) / len(samples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, default=Path("results/attention.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Attention performance measurements require CUDA.")
    dtype = getattr(torch, args.dtype)
    compiled = torch.compile(attention)
    rows = []
    for sequence_length, d in itertools.product(args.sequence_lengths, args.dimensions):
        for implementation, fn in (
            ("pytorch", attention),
            ("compiled", compiled),
            ("triton_flash", lambda q, k, v: FlashAttention2Triton.apply(q, k, v, True)),
        ):
            try:
                q, k, v = [
                    torch.randn(args.batch_size, sequence_length, d, device="cuda", dtype=dtype, requires_grad=True)
                    for _ in range(3)
                ]
                output = fn(q, k, v)
                memory_before_backward = torch.cuda.memory_allocated() / 1024**2
                grad = torch.randn_like(output)
                forward_ms = bench(lambda: fn(q, k, v), args.repetitions)
                backward_ms = bench(lambda: output.backward(grad, retain_graph=True), args.repetitions)
                rows.append({
                    "implementation": implementation,
                    "sequence_length": sequence_length,
                    "d": d,
                    "dtype": args.dtype,
                    "forward_ms": forward_ms,
                    "backward_ms": backward_ms,
                    "memory_before_backward_mib": memory_before_backward,
                    "status": "ok",
                })
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({
                    "implementation": implementation,
                    "sequence_length": sequence_length,
                    "d": d,
                    "dtype": args.dtype,
                    "forward_ms": "",
                    "backward_ms": "",
                    "memory_before_backward_mib": "",
                    "status": "oom",
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

