"""Hit rate versus layer. No model load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def readability_curve(readouts: pd.DataFrame, out: Path) -> pd.DataFrame:
    """``readouts`` needs ``layer``, ``lens``, ``k``, and ``hit``."""
    curve = readouts.groupby(["lens", "k", "layer"], as_index=False)["hit"].mean()
    out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out, index=False)
    return curve
