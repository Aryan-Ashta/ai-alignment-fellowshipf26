"""Write the depth table and plot from parquet results, when those files exist."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.analysis.depth import Thresholds  # noqa: E402
from s2.analysis.plots import plot_depth  # noqa: E402
from s2.analysis.tables import depth_table  # noqa: E402


def main() -> None:
    prereg = ROOT / "prereg" / "thresholds.yaml"
    if not prereg.is_file():
        raise SystemExit("prereg/thresholds.yaml does not exist yet; the pilot has not been frozen")
    thresholds = Thresholds.load(prereg)
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    frames = []
    for family in ("olmo3_7b", "olmo3_32b"):
        for slot, entry in models[family].items():
            root = ROOT / "results" / f"{family}.{slot}"
            nocot = root / "nocot.parquet"
            readouts = root / "readouts.parquet"
            swaps = root / "swaps.parquet"
            if not (nocot.is_file() and readouts.is_file() and swaps.is_file()):
                continue
            frames.append(
                {
                    "model": f"{family}.{slot}",
                    "revision": entry["revision"],
                    "rl_step": entry["rl_step"],
                    "nocot": pd.read_parquet(nocot),
                    "readouts": pd.read_parquet(readouts),
                    "swaps": pd.read_parquet(swaps),
                }
            )
    if not frames:
        raise SystemExit("no checkpoint has nocot, readout, and swap parquet yet")
    out = ROOT / "results" / "analysis"
    depth_table(frames, thresholds, out / "depth_by_checkpoint.csv")
    plot_depth(out / "depth_by_checkpoint.csv", out / "depth.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
