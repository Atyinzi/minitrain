"""Small byte-level BPE tokenizer used by MiniTrain inference."""

from __future__ import annotations

import pickle
from pathlib import Path

import regex


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.bytes_to_id = {token: token_id for token_id, token in self.vocab.items()}

    def _apply_bpe(self, piece: bytes) -> list[bytes]:
        parts = [bytes([value]) for value in piece]
        while len(parts) > 1:
            candidates = {
                (parts[index], parts[index + 1])
                for index in range(len(parts) - 1)
                if (parts[index], parts[index + 1]) in self.merge_ranks
            }
            if not candidates:
                break
            best = min(candidates, key=self.merge_ranks.get)
            merged = []
            index = 0
            while index < len(parts):
                if index + 1 < len(parts) and (parts[index], parts[index + 1]) == best:
                    merged.append(parts[index] + parts[index + 1])
                    index += 2
                else:
                    merged.append(parts[index])
                    index += 1
            parts = merged
        return parts

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if special_tokens:
            pattern = "|".join(regex.escape(token) for token in special_tokens)
            chunks = regex.split(f"({pattern})", text)
        else:
            chunks = [text]

        ids = []
        special_set = set(self.special_tokens)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:
                ids.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue
            for match in regex.finditer(GPT2_PRETOKEN_PATTERN, chunk):
                for token in self._apply_bpe(match.group().encode("utf-8")):
                    ids.append(self.bytes_to_id[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")

    @property
    def eos_token_id(self) -> int | None:
        return self.bytes_to_id.get(b"<|endoftext|>")


def load_tokenizer(path: Path) -> BPETokenizer:
    """Load the dictionary format produced by Assignment 1 BPE training."""
    with path.open("rb") as file:
        saved = pickle.load(file)
    if isinstance(saved, dict):
        return BPETokenizer(
            vocab=saved["vocab"],
            merges=saved["merges"],
            special_tokens=saved.get("special_tokens", []),
        )
    # Older workspaces may have pickled their Tokenizer instance directly.
    return BPETokenizer(saved.vocab, saved.merges, saved.special_tokens)
