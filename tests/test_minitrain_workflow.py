"""Exercise the public training and inference entry points on a tiny CPU model."""

import json
import math
import sys

import numpy as np
import pytest
import torch

from scripts import minitrain, minitrain_inference


def _train(monkeypatch, *, output, log_file, train_bin, val_bin, steps, resume=None):
    args = [
        "minitrain", "--device", "cpu", "--model-size", "tiny",
        "--vocab-size", "32", "--context-length", "8",
        "--batch-size", "2", "--steps", str(steps),
        "--train-bin", str(train_bin), "--val-bin", str(val_bin),
        "--eval-every", "1", "--eval-steps", "2",
        "--seed", "123", "--output", str(output), "--log-file", str(log_file),
    ]
    if resume is not None:
        args.extend(["--resume", str(resume)])
    monkeypatch.setattr(sys, "argv", args)
    minitrain.main()


def test_cpu_train_resume_and_infer(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    token_ids = (np.arange(128, dtype=np.uint16) % 32)
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    token_ids.tofile(train_bin)
    token_ids[::-1].tofile(val_bin)

    full = tmp_path / "full.pt"
    split = tmp_path / "split.pt"
    full_log = tmp_path / "full.jsonl"
    split_log = tmp_path / "split.jsonl"
    _train(monkeypatch, output=full, log_file=full_log, train_bin=train_bin,
           val_bin=val_bin, steps=2)
    _train(monkeypatch, output=split, log_file=split_log, train_bin=train_bin,
           val_bin=val_bin, steps=1)
    _train(monkeypatch, output=split, log_file=split_log, train_bin=train_bin,
           val_bin=val_bin, steps=1, resume=split)

    full_checkpoint = torch.load(full, map_location="cpu", weights_only=False)
    split_checkpoint = torch.load(split, map_location="cpu", weights_only=False)
    assert full_checkpoint["step"] == split_checkpoint["step"] == 2
    for name, expected in full_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(split_checkpoint["model_state_dict"][name], expected,
                                   rtol=0, atol=0)

    records = [json.loads(line) for line in split_log.read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert all(math.isfinite(record["val_loss"]) for record in records)

    report = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", [
        "minitrain_inference", "--checkpoint", str(split), "--device", "cpu",
        "--val-bin", str(val_bin), "--batch-size", "2", "--eval-steps", "2",
        "--output", str(report),
    ])
    minitrain_inference.main()
    evaluation = json.loads(report.read_text(encoding="utf-8"))
    assert evaluation["step"] == 2
    assert math.isfinite(evaluation["validation_loss"])
    assert evaluation["validation_loss"] == pytest.approx(records[-1]["val_loss"])
