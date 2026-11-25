"""Restricted J++ / R-lens fit. Refuses unless the laptop is on AC.

One prompt at a time. Accumulators are saved after every prompt so a killed
run resumes. Cotangents are ``g ⊙ w_v`` for readout rows and ``w_v`` for the
swap rows. This pass is the pooled R-lens; expert routing is a later pass
over the same VJP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.corpus import wikitext_long  # noqa: E402
from s2.fit_restricted import readout_cotangents, restricted_rows  # noqa: E402
from s2.models import load  # noqa: E402
from s2.runio import atomic_torch_save, code_commit, require_ac_power  # noqa: E402


def corpus(n: int) -> list[str]:
    return wikitext_long(n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--rows-per-pass", type=int, default=8)
    args = parser.parse_args()
    require_ac_power()
    key = f"{args.model}.{args.ckpt}"
    model = load(key, device=args.device)
    family = "olmo3" if args.model.startswith("olmo3") else "codi"
    candidates = json.loads((ROOT / "lenses" / family / "candidates.json").read_text())
    readout_ids = candidates["readout_ids"]
    swap_ids = candidates["rows"]["swap_nog"]
    head = model.lm_head.weight.detach().cpu()
    gain = model.final_norm.weight.detach().cpu()
    cotangents = torch.cat(
        [
            readout_cotangents(head, gain, readout_ids, include_gain=True),
            readout_cotangents(head, gain, swap_ids, include_gain=False),
        ],
        dim=0,
    )
    texts = corpus(args.prompts)
    out = ROOT / "lenses" / key
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "fit_accum.pt"
    start = 0
    acc = None
    if ckpt.is_file():
        saved = torch.load(ckpt, weights_only=True)
        if saved["code_commit"] != code_commit():
            raise SystemExit("fit checkpoint code commit differs; delete it to restart")
        start = int(saved["next_prompt"])
        acc = saved["acc"]
    arch = "olmo3" if family == "olmo3" else "llama"
    for index, text in enumerate(texts):
        if index < start:
            continue
        ids = model.encode_fit(text)
        if ids.shape[1] != 128:
            raise SystemExit(f"prompt {index} encoded to {ids.shape[1]} tokens, expected 128")
        rows = restricted_rows(model.hf, ids, cotangents.to(ids.device), arch=arch, rows_per_pass=args.rows_per_pass)
        acc = rows if acc is None else acc + rows
        atomic_torch_save({"acc": acc, "next_prompt": index + 1, "code_commit": code_commit()}, ckpt)
        print(f"prompt {index + 1}/{len(texts)}", flush=True)
    if acc is None:
        raise SystemExit("no prompts fitted")
    mean = acc / len(texts)
    atomic_torch_save(
        {
            "rlens_readout": mean[: len(readout_ids)],
            "swap_nog": mean[len(readout_ids) :],
            "readout_ids": readout_ids,
            "swap_ids": swap_ids,
        },
        out / "restricted_rows.pt",
    )
    print(f"wrote {out / 'restricted_rows.pt'}")


if __name__ == "__main__":
    main()
