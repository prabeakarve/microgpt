"""Text generation helpers built on top of a trained checkpoint."""

from __future__ import annotations

import numpy as np

from .model import GPT, softmax
from .tokenizer import ByteTokenizer


def complete(
    model: GPT,
    prompt: str,
    max_new_tokens: int = 240,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int | None = None,
) -> str:
    tok = ByteTokenizer()
    ids = tok.encode(prompt) or [10]
    idx = np.array([ids], dtype=np.int64)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return tok.decode(out[0])


def perplexity(model: GPT, text: str, stride: int | None = None) -> float:
    """Token-level perplexity of ``text`` under the model, scored in windows."""
    tok = ByteTokenizer()
    ids = np.array(tok.encode(text), dtype=np.int64)
    if len(ids) < 2:
        return float("nan")
    block = model.block_size
    stride = stride or block
    losses = []
    for start in range(0, len(ids) - 1, stride):
        chunk = ids[start : start + block + 1]
        if len(chunk) < 2:
            break
        x = chunk[None, :-1]
        y = chunk[None, 1:]
        loss, _ = model.loss_and_grads(x, y)
        losses.append(loss * (len(chunk) - 1))
    total = sum(losses)
    count = sum(len(ids[start : start + block + 1]) - 1 for start in range(0, len(ids) - 1, stride))
    return float(np.exp(total / max(count, 1)))


def next_token_distribution(model: GPT, context: str, top: int = 5):
    tok = ByteTokenizer()
    ids = np.array([tok.encode(context)[-model.block_size :]], dtype=np.int64)
    logits, _ = model.forward(ids)
    probs = softmax(logits[0, -1], axis=-1)
    order = np.argsort(probs)[::-1][:top]
    return [(tok.decode([int(i)]), float(probs[i])) for i in order]
