"""Lead-time table from per-position ranks. No model load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from s2.posthoc import lead_time


def lead_table(ranks: pd.DataFrame, out: Path, rank_max: int = 5) -> pd.DataFrame:
    """``ranks`` columns: item_id, step, position, rank, first_write."""
    rows = []
    for (item_id, step), block in ranks.groupby(["item_id", "step"]):
        ordered = block.sort_values("position")
        first = ordered["first_write"].iloc[0]
        first = None if pd.isna(first) else int(first)
        rows.append(
            {
                "item_id": item_id,
                "step": int(step),
                "lead": lead_time(ordered["rank"].tolist(), first, rank_max),
            }
        )
    table = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table
