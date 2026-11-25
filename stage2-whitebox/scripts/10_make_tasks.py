"""Build dev/pilot/main arithmetic parquets and data/MANIFEST.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import sha256_file  # noqa: E402
from s2.tasks import (  # noqa: E402
    MUL_SYMBOL,
    balance_report,
    compute_template_numbers,
    generate_items,
)

COLUMNS = [
    "item_id",
    "format",
    "split",
    "k",
    "x0",
    "ops",
    "operands",
    "values",
    "answer",
    "problem_text",
    "intermediates",
    "swap_b",
    "swap_yb",
    "patch_src_text",
    "patch_ctrl_text",
    "null_values",
]


def main() -> None:
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    tok = AutoTokenizer.from_pretrained(entry["repo"], revision=entry["revision"])
    template_numbers = sorted(compute_template_numbers(tok))
    print(f"TEMPLATE_NUMBERS {template_numbers}", flush=True)
    rows = generate_items(set(template_numbers), tok, MUL_SYMBOL)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fmt in ("nested", "word"):
        for split in ("dev", "pilot", "main"):
            part = frame[(frame["format"] == fmt) & (frame["split"] == split)]
            path = data / f"{fmt}_{split}.parquet"
            part.to_parquet(path, index=False)
            manifest[path.name] = sha256_file(path)
            print(f"{path.name} {len(part)} {manifest[path.name]}", flush=True)
    (data / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = {
        "mul_symbol": MUL_SYMBOL,
        "template_numbers": template_numbers,
        "nocot_prefill": None,
        "k_max": 6,
        "counts": {"dev": 50, "pilot": 200, "main": 300},
        "seeds": {"dev": 1, "pilot": 2, "main": 3},
    }
    (ROOT / "configs" / "tasks.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    lines = ["# Phase 1 histograms", "", "Balance gate (non-blocking): for k≥2, decile share in [0.06, 0.14] and op share in [0.25, 0.42].", ""]
    samples = ["# Phase 1 samples", "", "First 20 items of each (format, k) in the dev split, generation order.", ""]
    for fmt in ("nested", "word"):
        for split in ("dev", "pilot", "main"):
            for k in range(1, 7):
                part = frame[(frame["format"] == fmt) & (frame["split"] == split) & (frame["k"] == k)]
                chains = []
                from s2.tasks import Chain

                for row in part.itertuples(index=False):
                    chains.append(Chain(row.x0, tuple(row.ops), tuple(row.operands)))
                report = balance_report(chains)
                lines.append(
                    f"- {fmt} {split} k={k}: n={len(part)} gate={report['gate']} "
                    f"deciles={[round(s, 3) for s in report['decile_share']]} "
                    f"ops={ {op: round(s, 3) for op, s in report['op_share'].items()} }"
                )
                if split == "dev":
                    samples.append(f"## {fmt} k={k}")
                    samples.append("")
                    for text in part["problem_text"].head(20):
                        samples.append(f"- `{text}`")
                    samples.append("")
    (ROOT / "logs" / "phase1_histograms.md").write_text("\n".join(lines) + "\n")
    (ROOT / "logs" / "phase1_samples.md").write_text("\n".join(samples) + "\n")
    print("wrote data/MANIFEST.json")


if __name__ == "__main__":
    main()
