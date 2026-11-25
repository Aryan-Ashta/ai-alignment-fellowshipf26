"""Learn J++ mixing weights with jpp's optimizer. Arithmetic items are never passed in."""

from __future__ import annotations

import torch
from workspace_lens.fitting.condense_experts.fitter import (
    CondenseConfig,
    build_search_starting_points,
    fit_expert_weights,
)
from workspace_lens.fitting.condense_experts.rank_objective import build_pair_table


def learn_weights(logits_pev: torch.Tensor, candidate_ids: list[dict[str, list[int]]], eval_slugs: list[str]) -> torch.Tensor:
    """``logits_pev`` is ``[P, E, V]``. Returns unit-L2 weights ``[E]``."""
    pairs = build_pair_table(candidate_ids, eval_slugs, list(range(len(eval_slugs))), device=logits_pev.device)
    scores = torch.zeros(logits_pev.shape[1])
    starts = build_search_starting_points(scores, top_n=min(3, int(scores.numel())))
    learned = fit_expert_weights(
        logits_pev,
        pairs,
        excluded_mask_Bool_V=None,
        initial_weightings=starts,
        condense_config=CondenseConfig(steps=2, exact_every=1, pass_k=1),
    )
    weights = learned.weights_E
    return weights / weights.norm()
