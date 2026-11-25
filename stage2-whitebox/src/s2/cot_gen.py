"""CoT generation helpers. Sampling settings match Olmo's recommended 0.6 / 0.95."""

from __future__ import annotations

import re

_INTEGER = re.compile(r"\d+")


def answer_after_think(text: str) -> int | None:
    """Last integer after ``</think>``. None if the tag or an integer is missing."""
    marker = "</think>"
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    matches = _INTEGER.findall(tail)
    return int(matches[-1]) if matches else None
