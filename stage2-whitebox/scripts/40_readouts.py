"""No-CoT readouts at the last four positions. Refuses unless on AC.

Requires ``lenses/<model>.<ckpt>/restricted_rows.pt`` from ``31_fit.py``.
``rlens_readout`` is ``[C_read, L-1, d]``. Layer L-1 uses the identity transport.
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
from s2.readout import number_scores, rank_among  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402
from s2.tasks import render_prompt  # noqa: E402


def number_rows_at(model, rlens: torch.Tensor, layer: int) -> torch.Tensor:
    if layer < rlens.shape[1]:
        return rlens[:100, layer]
    return model.lm_head.weight[model.number_ids].detach().cpu().float() * model.final_norm.weight.detach().cpu().float()


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
    lens_path = ROOT / "lenses" / key / "restricted_rows.pt"
    if not lens_path.is_file():
        raise SystemExit(f"missing {lens_path}; run scripts/31_fit.py first")
    rlens = torch.load(lens_path, weights_only=True)["rlens_readout"]
    model = load(key, device=args.device)
    prefill = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["nocot_prefill"]
    items = pd.read_parquet(ROOT / "data" / f"{args.format}_{args.split}.parquet")
    number_index = {token_id: i for i, token_id in enumerate(model.number_ids)}
    rows = []
    for item in items.itertuples(index=False):
        _text, ids = render_prompt(model.tok, item.problem_text, "nocot", prefill)
        input_ids = torch.tensor([ids], device=args.device)
        with model.capture(list(range(model.n_layers))) as stored:
            logits = model.hf(input_ids=input_ids).logits
        targets = [("answer", -1, int(item.answer))]
        targets += [("intermediate", step + 1, int(value)) for step, value in enumerate(item.intermediates)]
        for layer in range(model.n_layers):
            hidden = stored[layer][0, -4:]
            lens_rows = number_rows_at(model, rlens, layer)
            for offset, position in enumerate(range(-4, 0)):
                num_scores = number_scores(hidden[offset].cpu(), lens_rows)
                logit_scores = logits[0, position].cpu()
                for kind, step, value in targets:
                    token_id = model.number_ids[value]
                    idx = number_index[token_id]
                    rows.append(
                        {
                            "item_id": item.item_id,
                            "k": int(item.k),
                            "format": args.format,
                            "target_kind": kind,
                            "step": step,
                            "value": value,
                            "layer": layer,
                            "pos_offset": position,
                            "lens": "rlens",
                            "rank_num": rank_among(num_scores, idx),
                            "score": float(num_scores[idx]),
                            "rank_vocab": None,
                        }
                    )
                    rows.append(
                        {
                            "item_id": item.item_id,
                            "k": int(item.k),
                            "format": args.format,
                            "target_kind": kind,
                            "step": step,
                            "value": value,
                            "layer": layer,
                            "pos_offset": position,
                            "lens": "logit",
                            "rank_num": rank_among(logit_scores[model.number_ids], idx),
                            "score": float(logit_scores[token_id]),
                            "rank_vocab": rank_among(logit_scores, token_id),
                        }
                    )
    out = ROOT / "results" / key / "readouts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
