"""Pilot report from artifacts that exist. Missing files stay missing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _rate(path: Path, column: str) -> str:
    if not path.is_file():
        return "missing"
    frame = pd.read_parquet(path)
    if frame.empty or column not in frame.columns:
        return "missing"
    return f"{float(frame[column].astype(float).mean()):.3f} n={len(frame)}"


def write_report(root: Path, key: str, out: Path) -> str:
    results = root / "results" / key
    lines = [
        f"# Pilot report: {key}",
        "",
        "Thresholds are not set here. Approve or change them before the main study.",
        "",
        f"- no-CoT accuracy: {_rate(results / 'nocot.parquet', 'correct')}",
        f"- leakage: {_rate(results / 'nocot.parquet', 'leaked')}",
        f"- swap success: {_rate(results / 'swaps.parquet', 'success')}",
        f"- patch success: {_rate(results / 'patch.parquet', 'success')}",
        "",
        "Stop. Do not start the main 7B study until these numbers are approved.",
        "",
    ]
    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text
