"""Accuracy and leakage tables. Bootstrap lives with the depth metric."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def accuracy_table(nocot: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for (fmt, k), block in nocot.groupby(["format", "k"]):
        leaked = block["leaked"].astype(bool)
        kept = block[~leaked]
        rows.append(
            {
                "format": fmt,
                "k": int(k),
                "n": len(block),
                "accuracy_all": float(block["correct"].astype(bool).mean()),
                "accuracy_unleaked": float(kept["correct"].astype(bool).mean()) if len(kept) else float("nan"),
                "leakage": float(leaked.mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["format", "k"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table
