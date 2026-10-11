"""Run and summarize the MiniTrain multi-GPU strategy benchmark suite.

The runner intentionally launches one strategy at a time so every result gets
the same number of GPUs and the same model configuration. Use
``--summarize-only`` to rebuild the report without launching GPU jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_parallel.py"
DEFAULT_STRATEGIES = (
    "naive",
    "flat",
    "overlap",
    "sharded_optimizer",
    "fsdp_no_prefetch",
    "fsdp",
)


def build_markdown_report(results: list[dict]) -> str:
    """Create a compact comparison table from benchmark JSON records."""
    by_strategy = {result["strategy"]: result for result in results}
    ordered = [by_strategy[name] for name in DEFAULT_STRATEGIES if name in by_strategy]
    ordered.extend(result for result in results if result["strategy"] not in DEFAULT_STRATEGIES)
    if not ordered:
        raise ValueError("没有可汇总的 benchmark 结果")

    baseline_ms = by_strategy.get("naive", ordered[0])["iteration_mean_ms"]
    lines = [
        "# MiniTrain Parallel Benchmark",
        "",
        "| Strategy | Step (ms) | Speedup | Tokens/s | Exposed comm (ms) | Comm % | Peak MiB | Params/rank MiB | Optimizer/rank MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in ordered:
        parameter_mib = max(result["parameter_storage_mib_per_rank"])
        optimizer_mib = max(result["optimizer_storage_mib_per_rank"])
        lines.append(
            "| {strategy} | {step:.2f} | {speedup:.2f}x | {tokens:.0f} | "
            "{communication:.2f} | {fraction:.1%} | {peak:.0f} | {parameters:.0f} | {optimizer:.0f} |".format(
                strategy=result["strategy"],
                step=result["iteration_mean_ms"],
                speedup=baseline_ms / result["iteration_mean_ms"],
                tokens=result["tokens_per_second"],
                communication=result["exposed_communication_mean_ms"],
                fraction=result["exposed_communication_fraction"],
                peak=result["peak_allocated_mib"],
                parameters=parameter_mib,
                optimizer=optimizer_mib,
            )
        )

    config = ordered[0]["configuration"]
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Model: {config['model_size']}",
            f"- Context length: {config['context_length']}",
            f"- Global batch size: {config['global_batch_size']}",
            f"- World size: {config['world_size']}",
            f"- Device: {config['device']}",
            "",
            "`Exposed comm` is communication time still visible after backward; lower is better. "
            "FSDP prefetch can trade a small amount of peak memory for less exposed communication.",
            "",
        ]
    )
    return "\n".join(lines)


def load_results(output_dir: Path, strategies: list[str]) -> list[dict]:
    results = []
    for strategy in strategies:
        path = output_dir / f"parallel_{strategy}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def write_summary(output_dir: Path, results: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parallel_summary.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "parallel_summary.md"
    report_path.write_text(build_markdown_report(results), encoding="utf-8")
    print(f"wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MiniTrain multi-GPU benchmarks")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2, help="Global batch size")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/parallel_suite"))
    parser.add_argument("--strategies", nargs="+", choices=DEFAULT_STRATEGIES, default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if not args.summarize_only:
        if args.nproc_per_node < 2:
            raise ValueError("并行策略对比至少需要 2 个进程")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus < args.nproc_per_node:
            raise RuntimeError(
                f"需要 {args.nproc_per_node} 张 GPU，但当前只检测到 {visible_gpus} 张；"
                "请在多 GPU 环境运行，或使用 --summarize-only 汇总已有结果"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for strategy in args.strategies:
            output_path = output_dir / f"parallel_{strategy}.json"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.nproc_per_node}",
                str(BENCHMARK_SCRIPT),
                "--strategy",
                strategy,
                "--model-size",
                args.model_size,
                "--context-length",
                str(args.context_length),
                "--batch-size",
                str(args.batch_size),
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--output",
                str(output_path),
            ]
            print(f"running {strategy} ...", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    results = load_results(output_dir, args.strategies)
    if not results:
        raise FileNotFoundError(f"在 {output_dir} 中没有找到并行 benchmark JSON")
    write_summary(output_dir, results)


if __name__ == "__main__":
    main()
