"""Evaluate a MiniTrain checkpoint and generate text from a prompt."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.tokenizer import BPETokenizer, load_tokenizer


def resolve_device(name: str) -> torch.device:
    if name == "gpu":
        name = "cuda"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 GPU")
    return device


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    attention_backend: str | None = None,
) -> tuple[BasicsTransformerLM, dict]:
    # Training checkpoints also contain large AdamW states. Keep the checkpoint
    # on CPU and copy only model weights to the requested inference device.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_config 或 model_state_dict")

    config = dict(checkpoint["model_config"])
    config["activation_checkpointing"] = False
    saved_backend = config.get("attention_backend", "sdpa")
    if attention_backend is not None:
        config["attention_backend"] = attention_backend
    elif device.type != "cuda" and saved_backend == "triton":
        config["attention_backend"] = "sdpa"

    model = BasicsTransformerLM(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "step": checkpoint.get("step"),
        "elapsed_time_seconds": checkpoint.get("elapsed_time_seconds"),
        "training_args": checkpoint.get("training_args", {}),
    }
    del checkpoint
    return model, metadata


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1]")

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities[remove] = 0
        probabilities = torch.zeros_like(probabilities)
        probabilities.scatter_(0, sorted_ids, sorted_probabilities)
        probabilities.div_(probabilities.sum())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def autocast_context(device: torch.device, mixed_precision: bool):
    if mixed_precision:
        if device.type != "cuda":
            raise ValueError("--mixed-precision 目前只支持 CUDA")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def generate(
    model: BasicsTransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    mixed_precision: bool,
    seed: int,
) -> tuple[str, list[int]]:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt 编码后不能为空")

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    for _ in range(max_new_tokens):
        context = token_ids[-model.context_length :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        with autocast_context(device, mixed_precision):
            logits = model(inputs)[0, -1]
        next_token = sample_next_token(logits, temperature, top_p, generator)
        token_ids.append(next_token)
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(token_ids), token_ids


@torch.no_grad()
def evaluate_validation_loss(
    model: BasicsTransformerLM,
    data: np.memmap,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    max_start = len(data) - model.context_length - 1
    if max_start < 0:
        raise ValueError("验证集 token 数量必须大于 context_length")
    starts = np.linspace(0, max_start, num=batch_size * eval_steps, dtype=np.int64)
    losses = []
    for step in range(eval_steps):
        step_starts = starts[step * batch_size : (step + 1) * batch_size]
        inputs = np.stack([data[start : start + model.context_length] for start in step_starts]).copy()
        targets = np.stack([data[start + 1 : start + model.context_length + 1] for start in step_starts]).copy()
        input_tensor = torch.from_numpy(inputs).long().to(device)
        target_tensor = torch.from_numpy(targets).long().to(device)
        with autocast_context(device, mixed_precision):
            logits = model(input_tensor)
            loss = F.cross_entropy(logits.flatten(0, 1), target_tensor.flatten())
        losses.append(float(loss.cpu()))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate or generate from a MiniTrain checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--val-bin", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--attention-backend", choices=("naive", "sdpa", "triton"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.prompt is None and args.val_bin is None:
        parser.error("至少提供 --prompt 或 --val-bin 中的一项")
    if args.prompt is not None and args.tokenizer is None:
        parser.error("使用 --prompt 时必须提供 --tokenizer")
    if args.batch_size < 1 or args.eval_steps < 1 or args.max_new_tokens < 0:
        parser.error("batch-size、eval-steps 必须为正数，max-new-tokens 不能为负数")

    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device, args.attention_backend)
    result = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint.get("step"),
        "device": str(device),
        "attention_backend": model.config["attention_backend"],
    }

    if args.val_bin is not None:
        validation_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
        result["validation_loss"] = evaluate_validation_loss(
            model,
            validation_data,
            args.batch_size,
            args.eval_steps,
            device,
            args.mixed_precision,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.prompt is not None:
        tokenizer = load_tokenizer(args.tokenizer)
        if len(tokenizer.vocab) != model.config["vocab_size"]:
            raise ValueError(
                f"tokenizer 词表大小为 {len(tokenizer.vocab)}，模型要求 {model.config['vocab_size']}"
            )
        text, token_ids = generate(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
            args.mixed_precision,
            args.seed,
        )
        print("\n=== Generated text ===\n")
        print(text)
        result["prompt"] = args.prompt
        result["generated_text"] = text
        result["token_count"] = len(token_ids)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
