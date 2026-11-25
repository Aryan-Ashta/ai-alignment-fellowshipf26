"""CoT accuracy from parsed answers. No model load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def cot_accuracy(records: pd.DataFrame, out: Path) -> pd.DataFrame:
    """``records`` needs ``k``, ``answer``, and ``parsed``."""
    frame = records.copy()
    frame["correct"] = frame["parsed"] == frame["answer"]
    table = frame.groupby("k", as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table
