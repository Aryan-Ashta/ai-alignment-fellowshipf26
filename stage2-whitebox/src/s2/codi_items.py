"""CODI Gate 4 prompts. Seed 4. No model load."""

from __future__ import annotations

import numpy as np

ADDITION = "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."
SUBTRACTION = "A team starts with {X} members. {Y} members leave the team. Then each remaining member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."


def gate4_items(n_per: int = 100, seed: int = 4) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for sign, template in (("+", ADDITION), ("-", SUBTRACTION)):
        got = 0
        while got < n_per:
            x, y, z = int(rng.integers(2, 10)), int(rng.integers(1, 10)), int(rng.integers(2, 6))
            s1 = x + y if sign == "+" else x - y
            if s1 < 1:
                continue
            s2 = s1 * z
            answer = s2 + s1
            if answer > 99 or len({x, y, z, s1, s2, answer}) < 6:
                continue
            out.append({"prompt": template.format(X=x, Y=y, Z=z), "x": x, "y": y, "z": z, "s1": s1, "s2": s2, "answer": answer, "sign": sign})
            got += 1
    return out
