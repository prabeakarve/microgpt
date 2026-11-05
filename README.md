# MicroGPT

**A GPT you can read end to end in one sitting — 700 lines of NumPy, no framework, no autograd.**

MicroGPT is a decoder-only Transformer language model implemented from first principles.
Every gradient is derived by hand, every shape is annotated, and the whole thing trains on a
laptop CPU in about a minute. It exists for one reason: to make the machinery of modern
language models legible instead of magical.

```
                      ┌──────────────────────────────────────────────┐
   token ids  ──────▶ │  tok_emb[idx] + pos_emb[:T]                  │
                      └───────────────────┬──────────────────────────┘
                                          ▼
              ┌───────────────────────────────────────────────┐
              │  × n_layer                                    │
              │   ┌─────────────────────────────────────────┐ │
              │   │ x = x + Attention(LN(x))                │ │
              │   │ x = x + MLP(LN(x))                      │ │
              │   └─────────────────────────────────────────┘ │
              └───────────────────┬───────────────────────────┘
                                  ▼
                      ┌───────────────────────────┐
                      │  LN_f → logits = x @ Eᵀ   │  (weight tying)
                      └───────────────────────────┘
```

## Why another tiny GPT

Most "build GPT from scratch" projects end up re-implementing an autograd engine, which hides
the interesting part behind a second abstraction. MicroGPT keeps the forward pass explicit and
writes the backward pass by hand, so the chain rule is visible in the code that uses it. The
test suite proves the backward pass correct with central finite differences against every
parameter class — attention projections, both layer norms, the MLP, and the tied embeddings.

## Features

- **Pure NumPy.** One dependency (`numpy`). Runs anywhere Python runs, including CI containers.
- **Hand-written backward pass.** No autograd, no `torch.autograd`, no graph object.
- **Causal multi-head self-attention** with residual connections and pre-layer-norm.
- **Weight tying** between the input embedding and the output projection.
- **Byte-level tokenizer** — 256 symbols, no training, exact UTF-8 round-trip.
- **AdamW** with bias correction, linear warmup, and cosine decay.
- **Sampling** with temperature, top-k, and a next-token distribution inspector.
- **Checkpointing** to compressed `.npz` plus a JSON sidecar with the full config and metrics.
- **14 tests**, including a gradient check with a `2e-4` relative-error bound.
- **CLI** with `train`, `sample`, `eval`, `info`, and `next` subcommands.

## Install

```bash
git clone https://github.com/prabeakarve/microgpt.git
cd microgpt
python -m pip install -e .
```

Python 3.9+ and NumPy are the only requirements.

## Quick start

Train on the bundled corpus (about 8 KB of text, ~60 seconds on a laptop):

```bash
python -m microgpt train \
  --corpus examples/corpus.txt \
  --out checkpoints/model.npz \
  --block-size 96 --n-layer 3 --n-head 4 --n-embd 128 \
  --batch-size 16 --steps 1200 --lr 3e-3 --warmup 60
```

Sample from the checkpoint:

```bash
python -m microgpt sample --ckpt checkpoints/model.npz \
  --prompt "A language model is a probability distribution over" \
  --tokens 200 --temperature 0.7 --top-k 40
```

Inspect what the model thinks comes next:

```bash
python -m microgpt next --context "Attention is a lookup that"
```

Score a text:

```bash
python -m microgpt eval --text examples/corpus.txt
```

## Library use

```python
import numpy as np
from microgpt import GPT, ByteTokenizer, TrainConfig, train, complete, perplexity

cfg = TrainConfig(corpus="examples/corpus.txt", steps=400, n_embd=96, n_layer=2)
meta = train(cfg)                       # writes cfg.out and cfg.out + ".json"

model = GPT(vocab_size=256, block_size=64, n_layer=2, n_head=4, n_embd=96)
loss, grads = model.loss_and_grads(idx, targets)   # idx, targets: int64 (B, T)

print(complete(model, "The model learns", max_new_tokens=120))
print(perplexity(model, open("examples/corpus.txt").read()))
```

## What the numbers mean

| Symbol | Meaning |
| --- | --- |
| `B` | batch size |
| `T` | sequence length (`<= block_size`) |
| `C` | embedding width (`n_embd`) |
| `H` | number of heads, `head_dim = C / H` |
| `V` | vocabulary size (256, byte-level) |

Cross-entropy loss is reported in nats per token. `ln(256) ≈ 5.545` is the loss of a model that
guesses uniformly, so any loss below that is real signal. Perplexity is `exp(loss)`.

## Design notes

**Why pre-layer-norm.** Normalizing before each sub-block keeps the residual stream's scale
stable, which is what makes deep stacks trainable without learning-rate tricks. The residual
path is a clean identity from the embedding to the final norm.

**Why scale the output projections.** `wo` and `w2` are multiplied by `1/sqrt(2 * n_layer)` at
initialization. Residual branches accumulate variance with depth; this keeps the stream's
magnitude roughly constant and prevents the first few dozen steps from diverging.

**Why the mask is applied before the softmax.** Setting masked scores to `-inf` makes their
post-softmax probability exactly zero, so no future token can influence the present. Applying a
mask afterwards would break the normalization.

**Why byte-level tokens.** A learned subword vocabulary is a second thing to train, version, and
get wrong. Bytes cost sequence length but remove an entire class of bugs, and they let the model
round-trip any UTF-8 text exactly — including text it has never seen.

## Project layout

```
microgpt/
  model.py        # Transformer: forward pass, hand-written backward pass, sampling
  optim.py        # AdamW and the warmup + cosine schedule
  train.py        # batching, training loop, checkpoint save/load
  generate.py     # completion, perplexity, next-token inspection
  tokenizer.py    # byte-level tokenizer
  __main__.py     # CLI
examples/
  corpus.txt      # ~8 KB training corpus
tests/
  test_microgpt.py
```

## Testing

```bash
python -m unittest discover -s tests -v
```

The gradient check compares analytic gradients against central differences for the token
embedding, positional embedding, query projection, output projection, both MLP matrices, and
both layer norms. A relative error above `2e-4` fails the build.

## 中文说明

MicroGPT 是一个**只用 NumPy 从零实现**的 GPT 语言模型：手写前向与反向传播，不依赖任何深度学习框架。

- **结构**：字节级分词（256 词表）+ 多头因果自注意力 + 残差 + Pre-LayerNorm + 4 倍扩展 FFN，输入输出嵌入权重共享。
- **训练**：交叉熵损失，AdamW 优化器，线性 warmup + 余弦退火，梯度用中心差分逐参数校验。
- **推理**：温度采样、top-k 截断、困惑度评估、下一个 token 概率分布查看。
- **上手**：`pip install -e .` 后执行 `python -m microgpt train`，一分钟内可在笔记本 CPU 上完成训练。

```bash
python -m microgpt train --steps 1200 --n-embd 128 --n-layer 3
python -m microgpt sample --prompt "A language model is" --tokens 200
```

适合用来理解注意力、反向传播、优化器和采样策略到底在做什么——代码量小到可以一个下午读完。

## License

MIT © 2026 prabeakarve
