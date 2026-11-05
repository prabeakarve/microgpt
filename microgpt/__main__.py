"""Command line entry point: ``python -m microgpt {train,sample,eval,info}``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .generate import complete, next_token_distribution, perplexity
from .train import TrainConfig, load_checkpoint, train


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", default="examples/corpus.txt")
    p.add_argument("--out", default="checkpoints/model.npz")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="microgpt",
        description="Train and sample a small GPT implemented in pure NumPy.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a model from a text corpus")
    _add_train_args(t)

    s = sub.add_parser("sample", help="generate text from a checkpoint")
    s.add_argument("--ckpt", default="checkpoints/model.npz")
    s.add_argument("--prompt", default="The model learns")
    s.add_argument("--tokens", type=int, default=240)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--seed", type=int, default=None)

    e = sub.add_parser("eval", help="report perplexity of a checkpoint on a file")
    e.add_argument("--ckpt", default="checkpoints/model.npz")
    e.add_argument("--text", default="examples/corpus.txt")

    i = sub.add_parser("info", help="print checkpoint metadata")
    i.add_argument("--ckpt", default="checkpoints/model.npz")

    n = sub.add_parser("next", help="show the next-token distribution for a context")
    n.add_argument("--ckpt", default="checkpoints/model.npz")
    n.add_argument("--context", default="The model learns")
    n.add_argument("--top", type=int, default=5)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "train":
        cfg = TrainConfig(
            corpus=args.corpus,
            out=args.out,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            warmup=args.warmup,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        train(cfg)
        return 0

    if args.cmd == "info":
        _, meta = load_checkpoint(args.ckpt)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    model, meta = load_checkpoint(args.ckpt)

    if args.cmd == "sample":
        text = complete(
            model,
            args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(text)
        return 0

    if args.cmd == "eval":
        with open(args.text, "r", encoding="utf-8") as fh:
            text = fh.read()
        print(f"perplexity: {perplexity(model, text):.3f}")
        return 0

    if args.cmd == "next":
        for piece, prob in next_token_distribution(model, args.context, top=args.top):
            shown = piece.replace("\n", "\\n").replace("\t", "\\t")
            print(f"{prob * 100:6.2f}%  {shown!r}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
