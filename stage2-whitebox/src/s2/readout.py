"""Ranks among the 100 number tokens. Ties favour the target."""

from __future__ import annotations

import torch


def rank_among(scores: torch.Tensor, target_index: int) -> int:
    """1-based rank of ``scores[target_index]``. Ties favour the target."""
    target = scores[target_index]
    return int(1 + (scores > target).sum())


def number_scores(hidden: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``hidden`` is ``[d]`` or ``[P, d]``. ``rows`` is ``[100, d]``. Returns scores over numbers."""
    return hidden.float() @ rows.float().T
