"""Single-node all-reduce benchmark for CS336 Assignment 2.

Implemented by Codex. Launch with torchrun so each rank owns one GPU.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[1, 10, 100, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/all_reduce.json"))
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    results = []
    for size_mib in args.sizes_mib:
        tensor = torch.ones(size_mib * 1024**2 // 4, device=device)
        for _ in range(args.warmup):
            dist.all_reduce(tensor)
        samples = []
        for _ in range(args.steps):
            torch.cuda.synchronize(device)
            start = timeit.default_timer()
            dist.all_reduce(tensor)
            torch.cuda.synchronize(device)
            samples.append((timeit.default_timer() - start) * 1_000)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, samples)
        if rank == 0:
            flat = [value for rank_samples in gathered for value in rank_samples]
            results.append({
                "world_size": dist.get_world_size(),
                "size_mib": size_mib,
                "mean_ms": statistics.mean(flat),
                "std_ms": statistics.stdev(flat),
            })
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

