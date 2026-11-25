"""CODI Gate 4 decision. Rates are fractions of correct items."""

from __future__ import annotations


def readability_passes(s1_rate: float, s2_rate: float, modal_s1: int | None, modal_s2: int | None) -> bool:
    if modal_s1 is None or modal_s2 is None:
        return False
    return s1_rate >= 0.60 and s2_rate >= 0.60 and modal_s1 < modal_s2


def patching_passes(source_rate: float, control_rate: float) -> bool:
    return source_rate >= 0.50 and control_rate <= 0.10
