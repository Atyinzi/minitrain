from scripts.run_parallel_suite import build_markdown_report


def _result(strategy: str, step_ms: float) -> dict:
    return {
        "strategy": strategy,
        "configuration": {
            "model_size": "small",
            "context_length": 128,
            "global_batch_size": 2,
            "world_size": 2,
            "device": "test GPU",
        },
        "iteration_mean_ms": step_ms,
        "tokens_per_second": 256 / (step_ms / 1_000),
        "exposed_communication_mean_ms": 10.0,
        "exposed_communication_fraction": 10.0 / step_ms,
        "peak_allocated_mib": 1000.0,
        "parameter_storage_mib_per_rank": [500.0, 500.0],
        "optimizer_storage_mib_per_rank": [900.0, 900.0],
    }


def test_build_markdown_report_calculates_speedup_and_order():
    report = build_markdown_report([_result("fsdp", 50.0), _result("naive", 100.0)])

    assert report.index("| naive |") < report.index("| fsdp |")
    assert "| naive | 100.00 | 1.00x |" in report
    assert "| fsdp | 50.00 | 2.00x |" in report
    assert "test GPU" in report
