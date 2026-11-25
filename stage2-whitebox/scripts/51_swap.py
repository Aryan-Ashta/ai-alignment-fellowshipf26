"""Lens swap on no-CoT-correct items. Refuses unless on AC.

Uses the g-free swap rows and the jpp clamp. Success is the final-position
top token equalling y_b's number token.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.models import load  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402
from s2.swaps import clamp_swap  # noqa: E402
from s2.tasks import render_prompt  # noqa: E402

SCALES = (0.5, 1.0, 2.0, 4.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--format", default="nested")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    require_ac_power()
    key = f"{args.model}.{args.ckpt}"
    lens = torch.load(ROOT / "lenses" / key / "restricted_rows.pt", weights_only=True)
    model = load(key, device=args.device)
    prefill = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["nocot_prefill"]
    nocot = pd.read_parquet(ROOT / "results" / key / "nocot.parquet")
    items = pd.read_parquet(ROOT / "data" / f"{args.format}_{args.split}.parquet")
    eligible = set(nocot[(nocot["correct"]) & (~nocot["leaked"])]["item_id"])
    rows = []
    vectors = lens["swap_nog"]
    for item in items.itertuples(index=False):
        if item.item_id not in eligible or item.k == 1 or not item.swap_b:
            continue
        a = int(item.intermediates[0])
        b = item.swap_b[0]
        yb = item.swap_yb[0]
        if b is None or yb is None:
            continue
        _text, ids = render_prompt(model.tok, item.problem_text, "nocot", prefill)
        input_ids = torch.tensor([ids], device=args.device)
        yb_id = model.number_ids[int(yb)]
        for layer in range(0, min(vectors.shape[1], model.n_layers - 1), 4):
            va = vectors[a, layer]
            vb = vectors[int(b), layer]
            if torch.nn.functional.cosine_similarity(va, vb, dim=0).abs() > 0.999:
                continue
            for scale in SCALES:
                def edit(h, va=va, vb=vb, scale=scale):
                    edited = h.clone()
                    edited[0, 1:] = clamp_swap(h[0, 1:].float(), va, vb, scale).to(h.dtype)
                    return edited
                with model.replace({layer: edit}):
                    top = int(model.hf(input_ids=input_ids).logits[0, -1].argmax())
                rows.append({"item_id": item.item_id, "k": int(item.k), "layer": layer, "scale": scale, "variant": "main", "success": top == yb_id})
    out = ROOT / "results" / key / "swaps.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
