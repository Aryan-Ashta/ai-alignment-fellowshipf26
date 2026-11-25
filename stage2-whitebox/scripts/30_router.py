"""Fit one router per checkpoint from block outputs. Refuses unless on AC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.corpus import wikitext_long  # noqa: E402
from s2.fit_restricted import kept_positions  # noqa: E402
from s2.kmeans_router import fit_and_save  # noqa: E402
from s2.models import load  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompts", type=int, default=32)
    args = parser.parse_args()
    require_ac_power()
    key = f"{args.model}.{args.ckpt}"
    model = load(key, device=args.device)
    texts = wikitext_long(args.prompts)
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(model.n_layers)}
    for text in texts:
        ids = model.encode_fit(text)
        with model.capture(list(range(model.n_layers))) as stored:
            model.hf(input_ids=ids)
        keep = kept_positions(ids.shape[1])
        for layer, hidden in stored.items():
            collected[layer].append(hidden[0, keep].detach().float().cpu())
    activations = {layer: torch.cat(rows, 0) for layer, rows in collected.items()}
    path = ROOT / "lenses" / key / "router.pt"
    fit_and_save(activations, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
