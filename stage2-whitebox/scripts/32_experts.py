"""Learn expert weights from a saved logit tensor. Does not invent the tensor.

Expected file keys: ``logits`` ``[P, E, V]``, ``candidate_ids``, ``eval_slugs``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.expert_weights import learn_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.logits.is_file():
        raise SystemExit(f"missing {args.logits}; macro-average logits have not been exported")
    saved = torch.load(args.logits, weights_only=True)
    weights = learn_weights(saved["logits"], saved["candidate_ids"], saved["eval_slugs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": weights, "experts": saved["eval_slugs"]}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
