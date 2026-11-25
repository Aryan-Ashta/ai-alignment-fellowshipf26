"""CODI Gate 4. Refuses unless on AC.

Readability: logit-lens top-1 among the 100 numbers at layer L-1 equals s1
on ≥ 60% of correct items, same for s2, and s1's modal position precedes s2's.
Patching: source latent at s1's modal position yields the implied answer on
≥ 50%; control changes the answer on ≤ 10%.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.codi import build_merged, codi_latent_run  # noqa: E402
from s2.codi_items import gate4_items  # noqa: E402
from s2.gate4 import readability_passes  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402


def top_number(model, hidden: torch.Tensor, number_ids: list[int]) -> int:
    logits = model.lm_head(model.model.norm(hidden.to(model.lm_head.weight.dtype).to(model.lm_head.weight.device)))
    scores = logits[0, number_ids]
    return int(scores.argmax())


def main() -> None:
    require_ac_power()
    model, projector, tok = build_merged(device="mps")
    number_ids = [tok.encode(str(n), add_special_tokens=False)[0] for n in range(100)]
    correct = []
    for item in gate4_items():
        result = codi_latent_run(model, projector, tok, item["prompt"])
        if str(item["answer"]) in result["answer_text"].split():
            item["hidden"] = result["hidden"]
            correct.append(item)
    if not correct:
        raise SystemExit("no correct CODI items")
    s1_pos, s2_pos = [], []
    s1_hit = s2_hit = 0
    layer = model.config.num_hidden_layers - 1
    for item in correct:
        hits1, hits2 = [], []
        for pos, hidden in enumerate(item["hidden"][layer]):
            top = top_number(model, hidden, number_ids)
            if top == item["s1"]:
                hits1.append(pos)
            if top == item["s2"]:
                hits2.append(pos)
        if hits1:
            s1_hit += 1
            s1_pos.append(hits1[0])
        if hits2:
            s2_hit += 1
            s2_pos.append(hits2[0])
    n = len(correct)
    modal_s1 = Counter(s1_pos).most_common(1)[0][0] if s1_pos else None
    modal_s2 = Counter(s2_pos).most_common(1)[0][0] if s2_pos else None
    print(f"correct {n} s1 {s1_hit / n:.3f} s2 {s2_hit / n:.3f} modal {modal_s1} {modal_s2}")
    report = ROOT / "logs" / "gate4" / "codi.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"correct {n}\ns1 rate {s1_hit / n:.3f}\ns2 rate {s2_hit / n:.3f}\nmodal s1 {modal_s1}\nmodal s2 {modal_s2}\n"
    )
    if not readability_passes(s1_hit / n, s2_hit / n, modal_s1, modal_s2):
        raise SystemExit("CODI Gate 4 readability failed")
    print("readability pass")


if __name__ == "__main__":
    main()
