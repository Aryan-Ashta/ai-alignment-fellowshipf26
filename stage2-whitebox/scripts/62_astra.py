"""Astra-like arm. Never invents the CoT-Control wording.

If ``configs/gen.yaml: astra_instruction`` is null, log the skip and exit 0.
"""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "AGENT_LOG.md"


def main() -> None:
    gen = yaml.safe_load((ROOT / "configs" / "gen.yaml").read_text()) or {}
    if not gen.get("astra_instruction"):
        line = "ASTRA ARM SKIPPED: wording unavailable\n"
        print(line, end="")
        with LOG.open("a") as handle:
            handle.write("\n" + line)
        return
    raise SystemExit("astra wording is set; the compliant-item readout is not launched from this script yet")


if __name__ == "__main__":
    main()
