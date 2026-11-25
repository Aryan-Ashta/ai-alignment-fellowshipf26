"""Greedy no-CoT accuracy. Refuses unless the laptop is on AC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.models import load  # noqa: E402
from s2.nocot import score_continuation  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402
from s2.tasks import render_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--format", default="nested")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    require_ac_power()
    model = load(f"{args.model}.{args.ckpt}", device=args.device)
    import yaml

    prefill = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["nocot_prefill"]
    items = pd.read_parquet(ROOT / "data" / f"{args.format}_{args.split}.parquet")
    if args.limit:
        items = items.head(args.limit)
    rows = []
    for row in items.itertuples(index=False):
        text, ids = render_prompt(model.tok, row.problem_text, "nocot", prefill)
        input_ids = torch.tensor([ids], device=args.device)
        generated = []
        with torch.no_grad():
            out = model.hf(input_ids=input_ids, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
            generated.append(nxt)
            cache = out.past_key_values
            for _ in range(7):
                out = model.hf(input_ids=torch.tensor([[nxt]], device=args.device), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
                generated.append(nxt)
        continuation = model.tok.decode(generated, skip_special_tokens=True)
        scored = score_continuation(continuation, int(row.answer), [int(v) for v in row.intermediates])
        rows.append({"item_id": row.item_id, "k": int(row.k), "format": args.format, "continuation": continuation, **scored})
        print(row.item_id, scored["correct"], flush=True)
    out = ROOT / "results" / f"{args.model}.{args.ckpt}" / "nocot.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
