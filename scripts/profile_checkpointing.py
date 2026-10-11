"""Activation-checkpoint block-size experiment for CS336 Assignment 2.

Implemented by Codex. It validates the sqrt(N) one-level checkpoint rule by
measuring neighboring segment sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from scripts.benchmark_transformer import MODEL_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/checkpointing.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint peak-memory profiling requires CUDA.")
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS["xl"]
    def run_segment(segment_size: int) -> dict[str, float]:
        model = BasicsTransformerLM(10_000, args.context_length, d_model, num_layers, num_heads, d_ff).cuda()

        def checkpointed_forward(tokens):
            x = model.token_embeddings(tokens)
            for start in range(0, len(model.layers), segment_size):
                layers = model.layers[start : start + segment_size]

                def segment(value, layers=layers):
                    for layer in layers:
                        value = layer(value)
                    return value

                x = checkpoint(segment, x, use_reentrant=False)
            return model.lm_head(model.ln_final(x))

        model.forward = checkpointed_forward
        tokens = torch.randint(10_000, (args.batch_size, args.context_length), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        model(tokens).sum().backward()
        torch.cuda.synchronize()
        return {"segment_size": segment_size, "peak_mib": torch.cuda.max_memory_allocated() / 1024**2}

    results = []
    for segment_size in args.segment_sizes:
        results.append(run_segment(segment_size))
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

