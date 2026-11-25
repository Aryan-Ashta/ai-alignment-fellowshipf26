"""Build the shared Olmo candidate rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.candidates import build_candidates, save_candidates  # noqa: E402


def main() -> None:
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    tok = AutoTokenizer.from_pretrained(entry["repo"], revision=entry["revision"])
    number_ids = json.loads((ROOT / "logs" / "preflight" / "number_token_ids_olmo3.json").read_text())
    ids = [number_ids["number_token_ids"][str(n)] for n in range(100)]
    payload = build_candidates(tok, ids, tok.vocab_size)
    path = ROOT / "lenses" / "olmo3" / "candidates.json"
    save_candidates(payload, path)
    print(f"C_read={payload['C_read']} C_total={payload['C_total']} -> {path}")


if __name__ == "__main__":
    main()
