"""Optimizers and a learning-rate schedule, in about one screen of code."""

from __future__ import annotations

import math

import numpy as np


class Adam:
    """Adam with decoupled weight decay (AdamW) and bias correction."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for name, param in params.items():
            g = grads[name]
            if param.ndim > 1 and self.wd:
                param -= lr * self.wd * param  # decoupled decay: no gradient mixing
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            mhat = self.m[name] / bc1
            vhat = self.v[name] / bc2
            param -= lr * mhat / (np.sqrt(vhat) + self.eps)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 50, min_ratio: float = 0.05) -> float:
    """Linear warmup followed by a cosine decay to ``base_lr * min_ratio``."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)
