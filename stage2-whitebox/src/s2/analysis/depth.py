"""Internal step depth from readout, no-CoT, and swap tables.

k=1 has no intermediates, so only the accuracy criterion applies. Depth is the
largest passing k and is not required to be monotone. The CI resamples items
within each k (10,000 draws, seed 0, percentile 95%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Thresholds:
    k_max: int
    formats: tuple[str, ...]
    t_acc: float
    lens: str
    rank_max: int
    positions: tuple[int, ...]
    layers: str
    min_item_rate: float
    t_share: float
    band_rule: str
    scale: float

    @classmethod
    def load(cls, path: str) -> "Thresholds":
        raw = yaml.safe_load(open(path))
        fmt = raw["format"]
        formats = ("nested", "word") if fmt == "both" else (fmt,)
        read = raw["T_read"]
        return cls(
            k_max=int(raw["K_max"]),
            formats=formats,
            t_acc=float(raw["T_acc"]),
            lens=read["lens"],
            rank_max=int(read["rank_max"]),
            positions=tuple(int(p) for p in read["positions"]),
            layers=str(read["layers"]),
            min_item_rate=float(read["min_item_rate"]),
            t_share=float(raw["T_share"]),
            band_rule=raw["swap"]["band_rule"],
            scale=float(raw["swap"]["scale"]),
        )


def _filter(frame: pd.DataFrame, formats: tuple[str, ...]) -> pd.DataFrame:
    if "format" in frame.columns:
        return frame[frame["format"].isin(formats)]
    return frame


def _readable(readouts: pd.DataFrame, thresholds: Thresholds, kind: str) -> pd.Series:
    part = readouts[
        (readouts["target_kind"] == kind)
        & (readouts["lens"] == thresholds.lens)
        & (readouts["pos_offset"].isin(thresholds.positions))
    ]
    if part.empty:
        return pd.Series(dtype=bool)
    best = part.groupby(["item_id", "step"])["rank_num"].min()
    return (best <= thresholds.rank_max).groupby(level=0).all()


def item_table(nocot: pd.DataFrame, readouts: pd.DataFrame, swaps: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    nocot = _filter(nocot, thresholds.formats)
    readouts = _filter(readouts, thresholds.formats)
    swaps = _filter(swaps, thresholds.formats)
    rows = []
    real = _readable(readouts, thresholds, "intermediate")
    null = _readable(readouts, thresholds, "null")
    main = swaps[(swaps["variant"] == "main") & (np.isclose(swaps["scale"], thresholds.scale))]
    swap_ok = main.groupby("item_id")["success"].mean() if not main.empty else pd.Series(dtype=float)
    for item_id, group in nocot.groupby("item_id"):
        row = group.iloc[0]
        rows.append(
            {
                "item_id": item_id,
                "k": int(row["k"]),
                "correct": bool(row["correct"]),
                "leaked": bool(row["leaked"]),
                "readable": bool(real.get(item_id, False)),
                "null_readable": bool(null.get(item_id, False)),
                "swap_success": float(swap_ok.get(item_id, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_p(real: np.ndarray, null: np.ndarray, n: int = 10_000, seed: int = 0) -> float:
    diff = real.astype(float) - null.astype(float)
    if len(diff) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float((means <= 0).mean())


def _k_passes(block: pd.DataFrame, thresholds: Thresholds) -> bool:
    kept = block[~block["leaked"]]
    acc = float(kept["correct"].mean()) if len(kept) else 0.0
    if acc <= thresholds.t_acc:
        return False
    k = int(block["k"].iloc[0])
    if k == 1:
        return True
    share = float(block["readable"].mean())
    if share < thresholds.min_item_rate:
        return False
    if paired_bootstrap_p(block["readable"].to_numpy(), block["null_readable"].to_numpy()) >= 0.05:
        return False
    if float(block["swap_success"].mean()) < thresholds.t_share:
        return False
    return True


def depth_from_items(items: pd.DataFrame, thresholds: Thresholds) -> int:
    passing = [int(k) for k, block in items.groupby("k") if _k_passes(block, thresholds)]
    return max(passing) if passing else 0


def depth_from_frames(nocot, readouts, swaps, thresholds) -> int:
    return depth_from_items(item_table(nocot, readouts, swaps, thresholds), thresholds)


def depth_ci(nocot, readouts, swaps, thresholds, n: int = 10_000, seed: int = 0) -> tuple[int, float, float]:
    items = item_table(nocot, readouts, swaps, thresholds)
    point = depth_from_items(items, thresholds)
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=int)
    groups = {int(k): block.reset_index(drop=True) for k, block in items.groupby("k")}
    for draw in range(n):
        parts = []
        for block in groups.values():
            take = block.iloc[rng.integers(0, len(block), size=len(block))]
            parts.append(take)
        samples[draw] = depth_from_items(pd.concat(parts, ignore_index=True), thresholds)
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, float(low), float(high)
