import numpy as np

from s2.tasks import (
    Chain,
    build_variants,
    constraints_ok,
    prompts_aligned,
    render_nested,
    render_word,
    select_balanced,
    draw_pool,
)


class Tok:
    """Stand-in that treats every character as its own token id."""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_nested_literal_and_values():
    chain = Chain(17, ("+", "-", "*"), (25, 9, 2))
    assert chain.values == (17, 42, 33, 66)
    assert chain.intermediates == (42, 33)
    assert chain.answer == 66
    assert render_nested(chain, "×") == "((17 + 25) - 9) × 2 ="
    assert render_word(chain) == "Start with 17. Add 25. Subtract 9. Multiply by 2. The result is"


def test_constraints_reject_collisions_and_template_numbers():
    ok = Chain(17, ("+", "-", "*"), (25, 9, 2))
    assert constraints_ok(ok, set())
    assert not constraints_ok(ok, {42})
    assert not constraints_ok(Chain(8, ("+", "+"), (8, 16)), set())
    assert not constraints_ok(Chain(6, ("+", "+", "-", "+"), (2, 2, 2, 1)), set())
    assert not constraints_ok(Chain(17, ("+",), (5,)), {22})


def test_k1_has_no_variants():
    chain = Chain(8, ("+",), (3,))
    variants = build_variants(chain, "nested", Tok(), set())
    assert variants == {
        "swap_b": [],
        "swap_yb": [],
        "patch_src_text": [],
        "patch_ctrl_text": [],
        "null_values": [],
    }


def test_alignment_requires_same_non_number_ids():
    tok = Tok()
    assert prompts_aligned(tok, "((17 + 25) - 9) × 2 =", "((18 + 24) - 8) × 3 =")
    assert not prompts_aligned(tok, "((17 + 25) - 9) × 2 =", "((17 + 25) - 9) * 2 =")


def test_draw_is_deterministic_and_balanced_selection_is_stable():
    banned: set[tuple] = set()
    first = draw_pool(np.random.default_rng(1001), 2, 40, set(), banned)
    second = draw_pool(np.random.default_rng(1001), 2, 40, set(), banned)
    assert [chain.key() for chain in first] == [chain.key() for chain in second]
    chosen = select_balanced(first, 8)
    again = select_balanced(second, 8)
    assert [chain.key() for chain in chosen] == [chain.key() for chain in again]
    assert len({chain.key() for chain in chosen}) == 8


def test_prompt_wrap_and_template_sha_match_across_7b():
    import yaml
    from pathlib import Path
    from transformers import AutoTokenizer
    from s2.tasks import chat_template_sha, render_prompt

    root = Path(__file__).resolve().parents[1]
    models = yaml.safe_load((root / "configs" / "models.yaml").read_text())
    shas = []
    tok = None
    for slot, entry in models["olmo3_7b"].items():
        tok = AutoTokenizer.from_pretrained(entry["repo"], revision=entry["revision"])
        shas.append(chat_template_sha(tok))
    assert len(set(shas)) == 1
    text, ids = render_prompt(tok, "((17 + 25) - 9) × 2 =", "nocot", "</think>\n\n")
    assert text.endswith("</think>\n\n")
    assert "<think>" in text
    assert ids[-1] == tok.encode("</think>\n\n", add_special_tokens=False)[-1]
    cot, _ = render_prompt(tok, "((17 + 25) - 9) × 2 =", "cot")
    assert "Answer with only the final number" not in cot
    assert cot.endswith("<think>\n") or cot.rstrip().endswith("<think>")
