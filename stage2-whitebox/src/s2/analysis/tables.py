"""SPEC §12 tables. One row per checkpoint for the depth table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from s2.analysis.depth import Thresholds, depth_ci


def depth_table(frames: list[dict], thresholds: Thresholds, out: Path) -> pd.DataFrame:
    rows = []
    for frame in frames:
        point, low, high = depth_ci(frame["nocot"], frame["readouts"], frame["swaps"], thresholds, n=200)
        rows.append(
            {
                "model": frame["model"],
                "revision": frame["revision"],
                "rl_step": frame["rl_step"],
                "depth": point,
                "ci_low": low,
                "ci_high": high,
            }
        )
    table = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table
