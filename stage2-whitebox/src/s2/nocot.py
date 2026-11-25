"""No-CoT answer parse and leakage flags.

Correct iff the first integer in the continuation equals the answer. A
continuation leaks when any intermediate appears as a standalone integer,
an operator from ``+ - * × =`` appears, more than one integer appears, or
the first non-space character is a letter.
"""

from __future__ import annotations

import re

_INTEGER = re.compile(r"\d+")
_OPERATOR = re.compile(r"[+\-*×=]")


def first_integer(text: str) -> int | None:
    match = _INTEGER.search(text)
    return int(match.group(0)) if match else None


def leakage_flags(continuation: str, intermediates: list[int]) -> dict[str, bool]:
    stripped = continuation.lstrip()
    integers = [int(match.group(0)) for match in _INTEGER.finditer(continuation)]
    flags = {
        "intermediate": any(value in integers for value in intermediates),
        "operator": _OPERATOR.search(continuation) is not None,
        "extra_integer": len(integers) > 1,
        "starts_with_letter": bool(stripped) and stripped[0].isalpha(),
    }
    flags["leaked"] = any(flags.values())
    return flags


def score_continuation(continuation: str, answer: int, intermediates: list[int]) -> dict:
    parsed = first_integer(continuation)
    return {"parsed": parsed, "correct": parsed == answer, **leakage_flags(continuation, intermediates)}
