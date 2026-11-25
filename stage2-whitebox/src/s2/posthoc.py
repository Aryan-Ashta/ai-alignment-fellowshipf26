"""Lead time of an intermediate relative to its first written number token."""

from __future__ import annotations


def lead_time(ranks: list[int], first_write: int | None, rank_max: int) -> int | None:
    """Positions by which the rank is already ≤ ``rank_max`` before the first write.

    ``ranks[i]`` is the rank at CoT position ``i``. ``None`` if it is never readable
    before the write, or never written.
    """
    if first_write is None:
        return None
    for index, rank in enumerate(ranks[:first_write]):
        if rank <= rank_max:
            return first_write - index
    return None
