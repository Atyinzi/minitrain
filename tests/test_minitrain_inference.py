import pickle

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.tokenizer import load_tokenizer
from scripts.minitrain_inference import load_model_checkpoint, sample_next_token


def test_load_tokenizer_dictionary_format(tmp_path):
    path = tmp_path / "tokenizer.pkl"
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"<|endoftext|>"
    with path.open("wb") as file:
        pickle.dump({"vocab": vocab, "merges": [], "special_tokens": ["<|endoftext|>"]}, file)

    tokenizer = load_tokenizer(path)

    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
    assert tokenizer.eos_token_id == 256


def test_checkpoint_config_reconstructs_model(tmp_path):
    config = {
        "vocab_size": 32,
        "context_length": 8,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
        "activation_checkpointing": False,
        "attention_backend": "sdpa",
    }
    original = BasicsTransformerLM(**config)
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state_dict": original.state_dict(), "step": 7}, path)

    restored, checkpoint = load_model_checkpoint(path, torch.device("cpu"))

    assert checkpoint["step"] == 7
    for expected, actual in zip(original.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_top_p_sampling_keeps_required_probability_mass():
    generator = torch.Generator().manual_seed(0)
    logits = torch.tensor([10.0, 1.0, 0.0])

    samples = {sample_next_token(logits, temperature=1.0, top_p=0.5, generator=generator) for _ in range(20)}

    assert samples == {0}
