"""Candidate rows for the restricted lens.

Order: number ids, fit-split dev targets, 500 distractors, then g-free swap
rows (the 100 numbers plus the 10 null-pool words). Readout rows use
cotangent ``g ⊙ w_v``. Swap rows use ``w_v`` without ``g``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lens_evals.causal_evals.probe_swap import _RANDOM_NULL_POOL
from lens_evals.eval_utils import candidate_token_ids, resolve_swap_token
from lens_evals.readout_evals.readout_eval_items import (
    MACRO_EVALS,
    load_readout_eval_items,
    split_fitting_items,
)
from lens_evals.readout_evals.readout_evals import READOUT_EVAL_SPECS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "third_party" / "jpp_lens" / "data" / "jlens" / "evaluations"


def _scoreable(tokenizer, items):
    kept = []
    for item in items:
        spec = READOUT_EVAL_SPECS[item.eval_slug]
        ok = any(
            candidate_token_ids(
                tokenizer,
                intermediate,
                expand_order_ops_synonyms=spec.expand_order_ops_synonyms,
            )
            for intermediate in item.intermediates
        )
        if ok:
            kept.append(item)
    return kept


def build_candidates(tokenizer, number_ids: list[int], vocab_size: int) -> dict:
    items = load_readout_eval_items(str(DATA_DIR), slugs=MACRO_EVALS)
    scoreable = _scoreable(tokenizer, items)
    fit, _held = split_fitting_items(scoreable, seed=0, fit_fraction=0.2)
    dev_ids: list[int] = []
    seen = set(number_ids)
    names: dict[str, list[str]] = {slug: [] for slug in MACRO_EVALS}
    for item in fit:
        names[item.eval_slug].append(item.name)
        spec = READOUT_EVAL_SPECS[item.eval_slug]
        for intermediate in item.intermediates:
            for token_id in candidate_token_ids(
                tokenizer,
                intermediate,
                expand_order_ops_synonyms=spec.expand_order_ops_synonyms,
            ):
                if token_id not in seen:
                    seen.add(token_id)
                    dev_ids.append(token_id)
    special = set(tokenizer.all_special_ids)
    added = set(getattr(tokenizer, "added_tokens_encoder", {}).values())
    pool = []
    for token_id in range(vocab_size):
        if token_id in seen or token_id in special or token_id in added:
            continue
        text = tokenizer.decode([token_id])
        stripped = text.strip()
        if not any(ch.isalnum() for ch in text):
            continue
        if stripped.isdigit():
            continue
        pool.append(token_id)
    rng = np.random.default_rng(0)
    take = min(500, len(pool))
    distractors = [int(pool[i]) for i in rng.choice(len(pool), size=take, replace=False)]
    distractors.sort()
    swap_words = []
    for word in _RANDOM_NULL_POOL:
        token_id, surface = resolve_swap_token(tokenizer, word)
        if token_id is None:
            raise RuntimeError(f"{word!r} has no single-token surface")
        swap_words.append({"word": word, "id": int(token_id), "surface": surface})
    rows = {
        "numbers": list(number_ids),
        "dev": dev_ids,
        "distractors": distractors,
        "swap_nog": list(number_ids) + [row["id"] for row in swap_words],
    }
    readout = rows["numbers"] + rows["dev"] + rows["distractors"]
    return {
        "fit_item_names": names,
        "swap_words": swap_words,
        "rows": rows,
        "C_read": len(readout),
        "C_total": len(readout) + len(rows["swap_nog"]),
        "readout_ids": readout,
    }


def save_candidates(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
