"""MicroGPT — a decoder-only Transformer language model written from scratch.

Everything here is plain NumPy: no autograd, no framework. The forward pass
returns the activations it needs, and the backward pass is written by hand.

Shapes are annotated as (B, T, C) = (batch, time, channels).
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray
DTYPE = np.float64


def _uniform(rng: np.random.Generator, fan_in: int, fan_out: int, gain: float = 1.0) -> Array:
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(DTYPE)


def _zeros(*shape: int) -> Array:
    return np.zeros(shape, dtype=DTYPE)


def _ones(*shape: int) -> Array:
    return np.ones(shape, dtype=DTYPE)


def gelu(x: Array) -> Array:
    """Gaussian error linear unit (tanh approximation)."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    return 0.5 * x * (1.0 + t)


def gelu_grad(x: Array, out: Array) -> Array:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * c * (1.0 + 3 * 0.044715 * x ** 2)


def softmax(x: Array, axis: int = -1) -> Array:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm_forward(x: Array, gamma: Array, beta: Array, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return xhat * gamma + beta, (xhat, inv, gamma)


def layer_norm_backward(dy: Array, cache, axis_axes):
    xhat, inv, gamma = cache
    reduce_axes = axis_axes
    dgamma = (dy * xhat).sum(axis=reduce_axes)
    dbeta = dy.sum(axis=reduce_axes)
    dxhat = dy * gamma
    m1 = dxhat.mean(axis=-1, keepdims=True)
    m2 = (dxhat * xhat).mean(axis=-1, keepdims=True)
    dx = inv * (dxhat - m1 - xhat * m2)
    return dx, dgamma, dbeta


class GPT:
    """A small causal Transformer with tied input/output embeddings."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_layer: int = 2,
        n_head: int = 4,
        n_embd: int = 64,
        seed: int = 0,
    ) -> None:
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.rng = np.random.default_rng(seed)

        rng = self.rng
        p: dict[str, Array] = {}
        p["tok_emb"] = rng.normal(0.0, 0.02, (vocab_size, n_embd)).astype(DTYPE)
        p["pos_emb"] = rng.normal(0.0, 0.01, (block_size, n_embd)).astype(DTYPE)
        for i in range(n_layer):
            k = f"b{i}."
            p[k + "ln1_g"] = _ones(n_embd)
            p[k + "ln1_b"] = _zeros(n_embd)
            p[k + "wq"] = _uniform(rng, n_embd, n_embd)
            p[k + "bq"] = _zeros(n_embd)
            p[k + "wk"] = _uniform(rng, n_embd, n_embd)
            p[k + "bk"] = _zeros(n_embd)
            p[k + "wv"] = _uniform(rng, n_embd, n_embd)
            p[k + "bv"] = _zeros(n_embd)
            p[k + "wo"] = _uniform(rng, n_embd, n_embd)
            p[k + "bo"] = _zeros(n_embd)
            p[k + "ln2_g"] = _ones(n_embd)
            p[k + "ln2_b"] = _zeros(n_embd)
            p[k + "w1"] = _uniform(rng, n_embd, 4 * n_embd)
            p[k + "b1"] = _zeros(4 * n_embd)
            p[k + "w2"] = _uniform(rng, 4 * n_embd, n_embd)
            p[k + "b2"] = _zeros(n_embd)
        p["lnf_g"] = _ones(n_embd)
        p["lnf_b"] = _zeros(n_embd)
        self.params = p
        # Scale the residual-writing projections down: keeps early training stable.
        scale = 1.0 / np.sqrt(2 * n_layer)
        for i in range(n_layer):
            self.params[f"b{i}.wo"] *= scale
            self.params[f"b{i}.w2"] *= scale

    # ---------------------------------------------------------------- forward
    def forward(self, idx: Array) -> tuple[Array, list]:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        p = self.params
        cache: dict = {"idx": idx, "T": T, "blocks": []}

        x = p["tok_emb"][idx] + p["pos_emb"][:T][None, :, :]  # (B,T,C)
        cache["x0"] = x

        for i in range(self.n_layer):
            k = f"b{i}."
            # --- attention sub-block
            h, ln1_cache = layer_norm_forward(x, p[k + "ln1_g"], p[k + "ln1_b"])
            q = h @ p[k + "wq"] + p[k + "bq"]
            kk = h @ p[k + "wk"] + p[k + "bk"]
            v = h @ p[k + "wv"] + p[k + "bv"]

            qh = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            kh = kk.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            vh = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

            scores = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)  # (B,H,T,T)
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -np.inf, scores)
            att = softmax(scores, axis=-1)
            ctx = att @ vh  # (B,H,T,D)
            ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.n_embd)
            attn_out = ctx @ p[k + "wo"] + p[k + "bo"]
            x = x + attn_out
            cache["blocks"].append(
                {"ln1": ln1_cache, "h": h, "q": q, "kk": kk, "v": v,
                 "att": att, "ctx": ctx, "x_pre": x - attn_out}
            )

            # --- MLP sub-block
            h2, ln2_cache = layer_norm_forward(x, p[k + "ln2_g"], p[k + "ln2_b"])
            a1 = h2 @ p[k + "w1"] + p[k + "b1"]
            g = gelu(a1)
            mlp_out = g @ p[k + "w2"] + p[k + "b2"]
            x = x + mlp_out
            cache["blocks"][-1].update(
                {"ln2": ln2_cache, "h2": h2, "a1": a1, "g": g, "x_mid": x - mlp_out}
            )

        xf, lnf_cache = layer_norm_forward(x, p["lnf_g"], p["lnf_b"])
        cache["lnf"] = lnf_cache
        cache["xf"] = xf
        logits = xf @ p["tok_emb"].T  # weight tying
        return logits, cache

    # ------------------------------------------------------------------ loss
    def loss_and_grads(self, idx: Array, targets: Array) -> tuple[float, dict]:
        B, T = idx.shape
        logits, cache = self.forward(idx)
        probs = softmax(logits, axis=-1)
        n = B * T
        eps = 1e-12
        loss = float(-np.log(probs.reshape(n, -1)[np.arange(n), targets.reshape(n)] + eps).mean())

        dlogits = probs.copy()
        dlogits.reshape(n, -1)[np.arange(n), targets.reshape(n)] -= 1.0
        dlogits /= n

        p = self.params
        grads: dict[str, Array] = {name: np.zeros_like(arr) for name, arr in p.items()}

        # lm_head is tied to tok_emb
        xf = cache["xf"].reshape(n, self.n_embd)
        dlogits2 = dlogits.reshape(n, -1)
        grads["tok_emb"] += dlogits2.T @ xf
        dxf = (dlogits2 @ p["tok_emb"]).reshape(B, T, self.n_embd)

        dx, dg, db = layer_norm_backward(dxf, cache["lnf"], (0, 1))
        grads["lnf_g"] = dg
        grads["lnf_b"] = db

        for i in reversed(range(self.n_layer)):
            k = f"b{i}."
            blk = cache["blocks"][i]
            # ---- MLP backward
            dx_mid = dx  # gradient flowing into the residual branch input
            dg_ = dx_mid.reshape(n, self.n_embd)
            g = blk["g"].reshape(n, 4 * self.n_embd)
            grads[k + "w2"] += g.T @ dg_
            grads[k + "b2"] += dg_.sum(axis=0)
            da1 = (dg_ @ p[k + "w2"].T) * gelu_grad(
                blk["a1"].reshape(n, 4 * self.n_embd), g
            )
            h2 = blk["h2"].reshape(n, self.n_embd)
            grads[k + "w1"] += h2.T @ da1
            grads[k + "b1"] += da1.sum(axis=0)
            dh2 = (da1 @ p[k + "w1"].T).reshape(B, T, self.n_embd)
            dx_ln2, dg2, db2 = layer_norm_backward(dh2, blk["ln2"], (0, 1))
            grads[k + "ln2_g"] = dg2
            grads[k + "ln2_b"] = db2
            dx = dx + dx_ln2  # residual: branch gradient plus the straight-through path

            # ---- attention backward
            dctx = dx.reshape(n, self.n_embd)
            grads[k + "wo"] += blk["ctx"].reshape(n, self.n_embd).T @ dctx
            grads[k + "bo"] += dctx.sum(axis=0)
            dctx2 = (dctx @ p[k + "wo"].T).reshape(B, T, self.n_head, self.head_dim)
            dctx2 = dctx2.transpose(0, 2, 1, 3)  # (B,H,T,D)
            att = blk["att"]
            vh = blk["v"].reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            datt = dctx2 @ vh.transpose(0, 1, 3, 2)
            dvh = att.transpose(0, 1, 3, 2) @ dctx2
            dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
            dqh = dscores @ blk["kk"].reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            dkh = dscores.transpose(0, 1, 3, 2) @ blk["q"].reshape(
                B, T, self.n_head, self.head_dim
            ).transpose(0, 2, 1, 3)
            dqh = dqh.transpose(0, 2, 1, 3).reshape(n, self.n_embd) / np.sqrt(self.head_dim)
            dkh = dkh.transpose(0, 2, 1, 3).reshape(n, self.n_embd) / np.sqrt(self.head_dim)
            dvh = dvh.transpose(0, 2, 1, 3).reshape(n, self.n_embd)
            h = blk["h"].reshape(n, self.n_embd)

            grads[k + "wq"] += h.T @ dqh
            grads[k + "bq"] += dqh.sum(axis=0)
            grads[k + "wk"] += h.T @ dkh
            grads[k + "bk"] += dkh.sum(axis=0)
            grads[k + "wv"] += h.T @ dvh
            grads[k + "bv"] += dvh.sum(axis=0)

            dh = dqh @ p[k + "wq"].T + dkh @ p[k + "wk"].T + dvh @ p[k + "wv"].T
            dx_ln1, dg1, db1 = layer_norm_backward(
                dh.reshape(B, T, self.n_embd), blk["ln1"], (0, 1)
            )
            grads[k + "ln1_g"] = dg1
            grads[k + "ln1_b"] = db1
            dx = dx + dx_ln1

        # embedding gradients
        np.add.at(grads["tok_emb"], cache["idx"].reshape(-1), dx.reshape(n, self.n_embd))
        grads["pos_emb"][:T] += dx.sum(axis=0)
        return loss, grads

    # -------------------------------------------------------------- sampling
    def generate(
        self,
        idx: Array,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        seed: int | None = None,
    ) -> Array:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        out = idx.copy()
        for _ in range(max_new_tokens):
            ctx = out[:, -self.block_size:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k < logits.shape[-1]:
                kth = np.partition(logits, -top_k, axis=-1)[:, -top_k]
                logits = np.where(logits < kth[:, None], -np.inf, logits)
            probs = softmax(logits, axis=-1)
            nxt = np.array(
                [[rng.choice(self.vocab_size, p=probs[b])] for b in range(probs.shape[0])]
            )
            out = np.concatenate([out, nxt], axis=1)
        return out

    # ------------------------------------------------------------ (de)serialize
    def state_dict(self) -> dict[str, Array]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: dict[str, Array]) -> None:
        missing = set(self.params) - set(state)
        if missing:
            raise KeyError(f"missing parameters: {sorted(missing)}")
        for k in self.params:
            self.params[k] = np.array(state[k], dtype=DTYPE)

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))
