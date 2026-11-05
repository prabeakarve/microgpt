"""Byte-level tokenizer.

A language model needs a vocabulary; the cheapest honest one is the set of
256 byte values. It never goes out of vocabulary, needs no training, and
round-trips any UTF-8 text exactly.
"""

from __future__ import annotations

import numpy as np


class ByteTokenizer:
    """Maps text to integers in [0, 255] and back."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.reshape(-1).tolist()
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ByteTokenizer(vocab_size={self.vocab_size})"
