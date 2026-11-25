"""Mean success by layer and scale. No smoothing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def mean_curve(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    curve = frame.groupby(["layer", "scale", "variant"], as_index=False)["success"].mean()
    out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out, index=False)
    return curve
