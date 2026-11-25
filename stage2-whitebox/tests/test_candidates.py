import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_rows_are_disjoint_and_counted():
    payload = json.loads((ROOT / "lenses" / "olmo3" / "candidates.json").read_text())
    numbers = payload["rows"]["numbers"]
    dev = payload["rows"]["dev"]
    distractors = payload["rows"]["distractors"]
    swap = payload["rows"]["swap_nog"]
    assert len(numbers) == 100
    assert len(distractors) == 500
    assert not set(numbers) & set(dev)
    assert not set(numbers) & set(distractors)
    assert not set(dev) & set(distractors)
    assert swap[:100] == numbers
    assert len(swap) == 110
    assert payload["C_read"] == len(numbers) + len(dev) + len(distractors)
    assert payload["C_total"] == payload["C_read"] + len(swap)
