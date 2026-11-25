"""Choose the shortest no-CoT prefill whose first token is a number on ≥ 90% of items.

50 dev k=1 nested items, 7B-DPO, greedy, 8 new tokens. Candidates, shortest first:
``</think>``, ``</think>\\n``, ``</think>\\n\\n``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.models import load  # noqa: E402
from s2.runio import require_ac_power, thermal_note, write_manifest  # noqa: E402
from s2.tasks import render_prompt  # noqa: E402

CANDIDATES = ("</think>", "</think>\n", "</think>\n\n")


def first_token_is_number(model, problem: str, prefill: str) -> bool:
    text, ids = render_prompt(model.tok, problem, "nocot", prefill)
    input_ids = torch.tensor([ids], device=next(model.hf.parameters()).device)
    with torch.no_grad():
        logits = model.hf(input_ids=input_ids).logits[0, -1]
    return int(logits.argmax()) in model.number_ids


def main() -> None:
    require_ac_power()
    before = thermal_note()
    model = load("olmo3_7b.dpo")
    items = pd.read_parquet(ROOT / "data" / "nested_dev.parquet")
    items = items[items["k"] == 1].head(50)
    rates = {}
    for prefill in CANDIDATES:
        hits = sum(first_token_is_number(model, text, prefill) for text in items["problem_text"])
        rates[prefill] = hits / len(items)
        print(f"{prefill!r} {rates[prefill]:.3f}", flush=True)
    chosen = next((p for p in CANDIDATES if rates[p] >= 0.90), None)
    if chosen is None:
        chosen = max(CANDIDATES, key=lambda p: (rates[p], -len(p)))
        if rates[chosen] < 0.50:
            raise SystemExit(f"best prefill {chosen!r} rate {rates[chosen]:.3f} < 0.50; stop and ask")
    config_path = ROOT / "configs" / "tasks.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["nocot_prefill"] = chosen
    config["nocot_prefill_rates"] = {p.replace("\n", "\\n"): rates[p] for p in CANDIDATES}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    lines = ["# Prefill calibration", "", f"Chosen `{chosen!r}` rate {rates[chosen]:.3f}.", ""]
    for prefill, rate in rates.items():
        lines.append(f"- `{prefill!r}`: {rate:.3f}")
    (ROOT / "logs" / "preflight" / "prefill.md").write_text("\n".join(lines) + "\n")
    write_manifest(
        ROOT / "logs" / "preflight",
        name="manifest_11_calibrate_prefill.json",
        script="11_calibrate_prefill.py",
        model_id=model.key,
        revision=model.revision,
        thermal_note={"before": before, "after": thermal_note()},
    )
    print(f"chose {chosen!r}")


if __name__ == "__main__":
    main()