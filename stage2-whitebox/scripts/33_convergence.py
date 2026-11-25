"""Compare a 16-prompt fit with a 32-prompt fit. Does not load a model.

Exit 0 when every readout row has cosine ≥ 0.99. Otherwise exit 3 and print
the minimum cosine. The 64-prompt fit is not started from this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.fit_restricted import prompts_converged  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.small, args.large):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    small = torch.load(args.small, weights_only=True)["rlens_readout"]
    large = torch.load(args.large, weights_only=True)["rlens_readout"]
    cos = torch.nn.functional.cosine_similarity(small.float(), large.float(), dim=-1)
    print(f"min cosine {float(cos.min()):.4f} rows {cos.numel()}")
    if not prompts_converged(small, large):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
