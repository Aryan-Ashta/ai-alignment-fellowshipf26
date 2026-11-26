"""Write the depth table and plot from parquet results, when those files exist."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.analysis.bootstrap import accuracy_table  # noqa: E402
from s2.analysis.cot_table import cot_accuracy  # noqa: E402
from s2.analysis.curves import mean_curve  # noqa: E402
from s2.analysis.readability import readability_curve  # noqa: E402
from s2.analysis.depth import Thresholds  # noqa: E402
from s2.analysis.plots import plot_depth  # noqa: E402
from s2.analysis.tables import depth_table  # noqa: E402


def main() -> None:
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    out = ROOT / "results" / "analysis"
    wrote_accuracy = False
    for family in ("olmo3_7b", "olmo3_32b"):
        for slot in models[family]:
            nocot = ROOT / "results" / f"{family}.{slot}" / "nocot.parquet"
            if nocot.is_file():
                accuracy_table(pd.read_parquet(nocot), out / f"accuracy_{family}.{slot}.csv")
                wrote_accuracy = True
            readouts = ROOT / "results" / f"{family}.{slot}" / "readouts.parquet"
            if readouts.is_file():
                frame = pd.read_parquet(readouts)
                if "hit" not in frame.columns and "rank_num" in frame.columns:
                    frame = frame.assign(hit=frame["rank_num"] == 0)
                if "hit" in frame.columns:
                    readability_curve(frame, out / f"readability_{family}.{slot}.csv")
                    wrote_accuracy = True
            root = ROOT / "results" / f"{family}.{slot}"
            for name in ("swaps", "patch"):
                path = root / f"{name}.parquet"
                if path.is_file():
                    mean_curve(pd.read_parquet(path), out / f"{name}_{family}.{slot}.csv")
                    wrote_accuracy = True
            cot = root / "cot" / "cot.jsonl"
            if cot.is_file() and cot.stat().st_size:
                records = pd.read_json(cot, lines=True)
                if {"k", "answer", "parsed"} <= set(records.columns):
                    cot_accuracy(records, out / f"cot_{family}.{slot}.csv")
                    wrote_accuracy = True
    prereg = ROOT / "prereg" / "thresholds.yaml"
    if not prereg.is_file():
        if wrote_accuracy:
            print(f"wrote accuracy tables under {out}; depth waits on prereg/thresholds.yaml")
            return
        raise SystemExit("prereg/thresholds.yaml does not exist yet; the pilot has not been frozen")
    thresholds = Thresholds.load(prereg)
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
        if wrote_accuracy:
            print(f"wrote accuracy tables under {out}; depth waits on readout and swap parquet")
            return
        raise SystemExit("no checkpoint has nocot, readout, and swap parquet yet")
    depth_table(frames, thresholds, out / "depth_by_checkpoint.csv")
    plot_depth(out / "depth_by_checkpoint.csv", out / "depth.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
