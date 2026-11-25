"""Merge CODI and check the reference subtraction prompt answers 5."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.codi import build_merged, codi_latent_run  # noqa: E402

PROMPT = (
    "A team starts with 3 members. 2 members leave the team. "
    "Then each remaining member recruits 4 additional people. "
    "How many people are there now on the team? Give the answer only and nothing else."
)


def main() -> None:
    model, projector, tok = build_merged(device="mps")
    out = ROOT / "scratch" / "codi_merged"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    result = codi_latent_run(model, projector, tok, PROMPT)
    print("answer:", result["answer_text"])
    print("ids:", result["answer_ids"])
    if "5" not in result["answer_text"].split():
        raise SystemExit(f"reference prompt did not answer 5: {result['answer_text']!r}")


if __name__ == "__main__":
    main()
