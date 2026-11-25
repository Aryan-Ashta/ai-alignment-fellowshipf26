"""Activation patching at the last four positions. Refuses unless on AC.

Success for the source patch is the top-1 next token equalling y_b's number token.
The control is reported as the flip rate away from y.
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
from s2.tasks import render_prompt  # noqa: E402


def capture_last4(model, text: str, prefill: str, layer: int) -> torch.Tensor:
    _rendered, ids = render_prompt(model.tok, text, "nocot", prefill)
    input_ids = torch.tensor([ids], device=next(model.hf.parameters()).device)
    with model.capture([layer]) as stored:
        model.hf(input_ids=input_ids)
    return stored[layer][0, -4:].detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--format", default="nested")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    require_ac_power()
    model = load(f"{args.model}.{args.ckpt}", device=args.device)
    prefill = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["nocot_prefill"]
    items = pd.read_parquet(ROOT / "data" / f"{args.format}_{args.split}.parquet")
    layers = list(range(0, model.n_layers - 1, 2 if model.n_layers <= 32 else 4))
    rows = []
    for item in items.itertuples(index=False):
        if item.k == 1 or not item.patch_src_text:
            continue
        src = item.patch_src_text[0]
        ctrl = item.patch_ctrl_text[0]
        yb = item.swap_yb[0] if item.swap_yb else None
        if src is None or yb is None:
            continue
        _text, ids = render_prompt(model.tok, item.problem_text, "nocot", prefill)
        input_ids = torch.tensor([ids], device=args.device)
        y_id = model.number_ids[int(item.answer)]
        yb_id = model.number_ids[int(yb)]
        for layer in layers:
            src_h = capture_last4(model, src, prefill, layer)
            def edit(h, src_h=src_h):
                out = h.clone()
                out[:, -4:, :] = src_h.to(h.device, h.dtype)
                return out
            with model.replace({layer: edit}):
                logits = model.hf(input_ids=input_ids).logits[0, -1]
            top = int(logits.argmax())
            rows.append({"item_id": item.item_id, "k": int(item.k), "layer": layer, "kind": "src", "success": top == yb_id, "top": top, "y": y_id, "yb": yb_id})
            if ctrl is None:
                continue
            ctrl_h = capture_last4(model, ctrl, prefill, layer)
            def edit_ctrl(h, ctrl_h=ctrl_h):
                out = h.clone()
                out[:, -4:, :] = ctrl_h.to(h.device, h.dtype)
                return out
            with model.replace({layer: edit_ctrl}):
                top_c = int(model.hf(input_ids=input_ids).logits[0, -1].argmax())
            rows.append({"item_id": item.item_id, "k": int(item.k), "layer": layer, "kind": "ctrl", "success": top_c != y_id, "top": top_c, "y": y_id, "yb": yb_id})
    out = ROOT / "results" / f"{args.model}.{args.ckpt}" / "patching.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
