"""Tests for MicroGPT. Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microgpt.model import GPT, layer_norm_backward, layer_norm_forward, softmax  # noqa: E402
from microgpt.optim import Adam, cosine_lr  # noqa: E402
from microgpt.tokenizer import ByteTokenizer  # noqa: E402
from microgpt.train import TrainConfig, load_checkpoint, train  # noqa: E402


def tiny_model(seed: int = 0) -> GPT:
    return GPT(vocab_size=256, block_size=12, n_layer=1, n_head=2, n_embd=16, seed=seed)


class TestTokenizer(unittest.TestCase):
    def test_roundtrip_ascii(self):
        tok = ByteTokenizer()
        text = "Attention is a lookup table."
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_roundtrip_unicode(self):
        tok = ByteTokenizer()
        text = "混合中文 and emoji 🚀 mixed"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_vocab_range(self):
        tok = ByteTokenizer()
        ids = tok.encode("Zoë")
        self.assertTrue(all(0 <= i < tok.vocab_size for i in ids))


class TestNumerics(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        x = np.random.default_rng(0).normal(size=(4, 7))
        p = softmax(x, axis=-1)
        np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)

    def test_layernorm_gradients(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 3, 5))
        gamma = rng.normal(size=5)
        beta = rng.normal(size=5)
        out, cache = layer_norm_forward(x, gamma, beta)
        dout = rng.normal(size=out.shape)
        dx, dgamma, dbeta = layer_norm_backward(dout, cache, (0, 1))

        eps = 1e-6
        for idx in [(0, 0, 2), (1, 2, 4)]:
            xp = x.copy()
            xp[idx] += eps
            lp = (layer_norm_forward(xp, gamma, beta)[0] * dout).sum()
            xm = x.copy()
            xm[idx] -= eps
            lm = (layer_norm_forward(xm, gamma, beta)[0] * dout).sum()
            numeric = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(numeric, dx[idx], places=6)


class TestModel(unittest.TestCase):
    def test_forward_shapes(self):
        m = tiny_model()
        idx = np.zeros((3, 7), dtype=np.int64)
        logits, cache = m.forward(idx)
        self.assertEqual(logits.shape, (3, 7, 256))
        self.assertEqual(len(cache["blocks"]), 1)

    def test_rejects_long_sequence(self):
        m = tiny_model()
        with self.assertRaises(ValueError):
            m.forward(np.zeros((1, 13), dtype=np.int64))

    def test_gradient_check(self):
        """Central finite differences against the hand written backward pass."""
        m = tiny_model(seed=7)
        rng = np.random.default_rng(11)
        idx = rng.integers(0, 256, size=(2, 8))
        tgt = rng.integers(0, 256, size=(2, 8))
        _, grads = m.loss_and_grads(idx, tgt)

        checked = [
            ("tok_emb", (5, 3)),
            ("pos_emb", (2, 1)),
            ("b0.wq", (0, 4)),
            ("b0.wo", (7, 9)),
            ("b0.w1", (3, 3)),
            ("b0.w2", (11, 2)),
            ("b0.ln1_g", (1,)),
            ("lnf_b", (6,)),
        ]
        eps = 1e-6
        for name, index in checked:
            param = m.params[name]
            original = param[index]
            param[index] = original + eps
            lp, _ = m.loss_and_grads(idx, tgt)
            param[index] = original - eps
            lm, _ = m.loss_and_grads(idx, tgt)
            param[index] = original
            numeric = (lp - lm) / (2 * eps)
            analytic = grads[name][index]
            denom = max(1e-8, abs(numeric) + abs(analytic))
            rel = abs(numeric - analytic) / denom
            self.assertLess(rel, 2e-4, f"{name}{index}: numeric={numeric:.8f} analytic={analytic:.8f}")

    def test_causal_mask_hides_future(self):
        m = tiny_model(seed=2)
        a = np.array([[10, 11, 12, 13]], dtype=np.int64)
        b = np.array([[10, 11, 12, 99]], dtype=np.int64)
        logits_a, _ = m.forward(a)
        logits_b, _ = m.forward(b)
        np.testing.assert_allclose(logits_a[0, :3], logits_b[0, :3], rtol=1e-10, atol=1e-10)

    def test_generate_length(self):
        m = tiny_model()
        out = m.generate(np.array([[65]], dtype=np.int64), max_new_tokens=5, seed=0)
        self.assertEqual(out.shape, (1, 6))

    def test_state_dict_roundtrip(self):
        a = tiny_model(seed=5)
        b = GPT(vocab_size=256, block_size=12, n_layer=1, n_head=2, n_embd=16, seed=9)
        idx = np.arange(6, dtype=np.int64)[None, :]
        before = a.forward(idx)[0]
        b.load_state_dict(a.state_dict())
        np.testing.assert_allclose(before, b.forward(idx)[0], rtol=1e-12)

    def test_loss_decreases(self):
        m = tiny_model(seed=1)
        rng = np.random.default_rng(0)
        opt = Adam(m.params, lr=1e-2)
        # A learnable pattern: the target token follows the input token.
        idx = rng.integers(0, 256, size=(4, 8))
        tgt = (idx + 1) % 256
        first = None
        for _ in range(40):
            loss, grads = m.loss_and_grads(idx, tgt)
            if first is None:
                first = loss
            opt.step(m.params, grads)
        self.assertLess(loss, first * 0.8)


class TestTrainingLoop(unittest.TestCase):
    def test_end_to_end_train_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ckpt.npz")
            cfg = TrainConfig(
                corpus="examples/corpus.txt",
                out=out,
                block_size=16,
                n_layer=1,
                n_head=2,
                n_embd=16,
                batch_size=4,
                steps=3,
                eval_every=2,
                warmup=1,
            )
            meta = train(cfg)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(meta["num_params"], 0)
            model, loaded = load_checkpoint(out)
            self.assertEqual(model.n_embd, 16)
            self.assertEqual(loaded["config"]["steps"], 3)

    def test_cosine_schedule_shape(self):
        self.assertLess(cosine_lr(0, 100, 1.0, warmup=10), cosine_lr(9, 100, 1.0, warmup=10))
        self.assertAlmostEqual(cosine_lr(100, 100, 1.0, warmup=10), 0.05, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
