"""Check a finished fit tensor against the candidate file. Does not load a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--hidden", type=int, default=4096)
    args = parser.parse_args()
    key = f"{args.model}.{args.ckpt}"
    path = ROOT / "lenses" / key / "restricted_rows.pt"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    saved = torch.load(path, weights_only=True)
    family = "olmo3" if args.model.startswith("olmo3") else "codi"
    candidates = json.loads((ROOT / "lenses" / family / "candidates.json").read_text())
    readout = saved["rlens_readout"]
    swap = saved["swap_nog"]
    if tuple(readout.shape) != (len(candidates["readout_ids"]), args.layers, args.hidden):
        raise SystemExit(f"readout shape {tuple(readout.shape)} != {(len(candidates['readout_ids']), args.layers, args.hidden)}")
    if tuple(swap.shape) != (len(candidates["rows"]["swap_nog"]), args.layers, args.hidden):
        raise SystemExit(f"swap shape {tuple(swap.shape)} != {(len(candidates['rows']['swap_nog']), args.layers, args.hidden)}")
    if saved["readout_ids"] != candidates["readout_ids"]:
        raise SystemExit("saved readout ids do not match candidates.json")
    print(f"ok {path} readout={tuple(readout.shape)} swap={tuple(swap.shape)}")


if __name__ == "__main__":
    main()
