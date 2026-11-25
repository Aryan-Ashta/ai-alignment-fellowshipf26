"""Which pilot artifacts are still missing. Does not invent thresholds."""

from __future__ import annotations

from pathlib import Path

STEPS = (
    ("fit", "lenses/{key}/restricted_rows.pt"),
    ("readouts", "results/{key}/readouts.parquet"),
    ("nocot", "results/{key}/nocot.parquet"),
    ("swaps", "results/{key}/swaps.parquet"),
    ("patch", "results/{key}/patch.parquet"),
)


def missing_steps(root: Path, key: str) -> list[str]:
    return [name for name, pattern in STEPS if not (root / pattern.format(key=key)).is_file()]
