"""Training loop: batching, the step function, checkpointing, and logging."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from .model import GPT
from .optim import Adam, cosine_lr
from .tokenizer import ByteTokenizer


@dataclass
class TrainConfig:
    corpus: str = "examples/corpus.txt"
    out: str = "checkpoints/model.npz"
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 96
    batch_size: int = 12
    steps: int = 300
    lr: float = 3e-3
    warmup: int = 40
    weight_decay: float = 0.01
    eval_every: int = 50
    eval_batches: int = 8
    seed: int = 1337


def load_corpus(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if not text.strip():
        raise ValueError(f"corpus {path!r} is empty")
    return np.array(ByteTokenizer().encode(text), dtype=np.int64)


def split(data: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    cut = max(1, int(n * (1.0 - val_frac)))
    return data[:cut], data[cut:]


def get_batch(data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator):
    if len(data) <= block_size + 1:
        raise ValueError("corpus too short for the requested block_size")
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts])
    y = np.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y


def _train(cfg: TrainConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    data = load_corpus(cfg.corpus)
    train_data, val_data = split(data)

    model = GPT(
        vocab_size=256,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        seed=cfg.seed,
    )
    opt = Adam(model.params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    history: list[dict] = []
    t0 = time.time()

    for step in range(cfg.steps):
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, rng)
        loss, grads = model.loss_and_grads(x, y)
        lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup)
        opt.step(model.params, grads, lr=lr)

        if step % 10 == 0 or step == cfg.steps - 1:
            print(f"step {step:4d} | loss {loss:.4f} | lr {lr:.2e} | {time.time() - t0:6.1f}s", flush=True)

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            val = evaluate(model, val_data, cfg, np.random.default_rng(cfg.seed + 1))
            history.append({"step": step + 1, "train_loss": loss, "val_loss": val})
            print(f"  eval @ {step + 1}: val_loss {val:.4f}", flush=True)

    meta = {
        "config": asdict(cfg),
        "vocab_size": 256,
        "num_params": model.num_params(),
        "history": history,
        "train_seconds": round(time.time() - t0, 2),
        "corpus_bytes": int(len(data)),
    }
    np.savez_compressed(cfg.out, **model.state_dict())
    with open(cfg.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved checkpoint -> {cfg.out}  ({model.num_params():,} params)")
    return meta


def evaluate(model: GPT, data: np.ndarray, cfg: TrainConfig, rng: np.random.Generator) -> float:
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.block_size, cfg.batch_size, rng)
        loss, _ = model.loss_and_grads(x, y)
        losses.append(loss)
    return float(np.mean(losses))


def train(cfg: TrainConfig) -> dict:
    return _train(cfg)


def load_checkpoint(path: str) -> tuple[GPT, dict]:
    blob = np.load(path)
    meta_path = path + ".json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    cfg = meta.get("config", {})
    model = GPT(
        vocab_size=256,
        block_size=cfg.get("block_size", 64),
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 96),
    )
    model.load_state_dict({k: blob[k] for k in blob.files})
    return model, meta
