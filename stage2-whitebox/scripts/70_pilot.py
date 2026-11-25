"""Pilot status. Does not invent thresholds or launch on battery.

Without ``--launch`` it prints the missing artifacts for the three pilot
checkpoints and exits 2 if any are missing. ``--launch`` refuses off AC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.pilot import missing_steps  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402

PILOT = ("olmo3_7b.base", "olmo3_7b.step_1375", "olmo3_7b.dpo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        require_ac_power()
        raise SystemExit("AC check passed; launch the per-checkpoint scripts, do not bypass their own AC checks")
    missing = {key: missing_steps(ROOT, key) for key in PILOT}
    for key, steps in missing.items():
        print(f"{key}: {', '.join(steps) if steps else 'complete'}")
    if any(missing.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
