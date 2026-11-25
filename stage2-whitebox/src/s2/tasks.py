"""Arithmetic chains for the Stage 2 no-CoT depth measurement.

Values stay in 0–99. Intermediates never appear in the prompt. The same
normalized chain is not reused across formats or splits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

OPS = ("+", "-", "*")
MUL_SYMBOL = "×"
WORD_OP = {"+": "Add {n}.", "-": "Subtract {n}.", "*": "Multiply by {n}."}


@dataclass(frozen=True)
class Chain:
    x0: int
    ops: tuple[str, ...]
    operands: tuple[int, ...]

    @property
    def k(self) -> int:
        return len(self.ops)

    @property
    def values(self) -> tuple[int, ...]:
        out = [self.x0]
        for op, n in zip(self.ops, self.operands, strict=True):
            out.append(apply_op(out[-1], op, n))
        return tuple(out)

    @property
    def intermediates(self) -> tuple[int, ...]:
        return self.values[1:-1]

    @property
    def answer(self) -> int:
        return self.values[-1]

    def key(self) -> tuple:
        return (self.x0, self.ops, self.operands)

    def item_id(self, fmt: str) -> str:
        steps = "-".join(f"{op}{n}" for op, n in zip(self.ops, self.operands, strict=True))
        return f"{fmt}-k{self.k}-{self.x0}-{steps}"


def apply_op(v: int, op: str, n: int) -> int:
    if op == "+":
        return v + n
    if op == "-":
        return v - n
    if op == "*":
        return v * n
    raise ValueError(op)


def legal_moves(v: int) -> list[tuple[str, int]]:
    moves: list[tuple[str, int]] = []
    for n in range(1, 99 - v + 1):
        moves.append(("+", n))
    for n in range(1, v + 1):
        moves.append(("-", n))
    for n in range(2, 10):
        if v * n <= 99:
            moves.append(("*", n))
    return moves


def render_nested(chain: Chain, mul_symbol: str = MUL_SYMBOL) -> str:
    symbols = {"+": "+", "-": "-", "*": mul_symbol}
    expr = "(" * (chain.k - 1) + str(chain.x0)
    for i, (op, n) in enumerate(zip(chain.ops, chain.operands, strict=True)):
        expr = f"{expr} {symbols[op]} {n}"
        if i < chain.k - 1:
            expr += ")"
    return expr + " ="


def render_word(chain: Chain) -> str:
    parts = [f"Start with {chain.x0}."]
    for op, n in zip(chain.ops, chain.operands, strict=True):
        parts.append(WORD_OP[op].format(n=n))
    parts.append("The result is")
    return " ".join(parts)


def render(chain: Chain, fmt: str, mul_symbol: str = MUL_SYMBOL) -> str:
    if fmt == "nested":
        return render_nested(chain, mul_symbol)
    if fmt == "word":
        return render_word(chain)
    raise ValueError(fmt)


def constraints_ok(chain: Chain, template_numbers: set[int]) -> bool:
    values = chain.values
    if any(not 0 <= v <= 99 for v in values):
        return False
    prompt_numbers = {chain.x0, *chain.operands}
    intermediates = chain.intermediates
    if len(set(intermediates)) != len(intermediates):
        return False
    banned = prompt_numbers | {chain.answer} | template_numbers
    if any(a in banned for a in intermediates):
        return False
    if chain.answer in template_numbers:
        return False
    return True


def hash64(*parts: object) -> int:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _sample_chain(rng: np.random.Generator, k: int, template_numbers: set[int]) -> Chain | None:
    x0 = int(rng.integers(1, 100))
    ops: list[str] = []
    operands: list[int] = []
    v = x0
    for _ in range(k):
        moves = legal_moves(v)
        if not moves:
            return None
        op, n = moves[int(rng.integers(0, len(moves)))]
        v = apply_op(v, op, n)
        ops.append(op)
        operands.append(n)
    chain = Chain(x0, tuple(ops), tuple(operands))
    if not constraints_ok(chain, template_numbers):
        return None
    return chain


def draw_pool(
    rng: np.random.Generator,
    k: int,
    count: int,
    template_numbers: set[int],
    banned_keys: set[tuple],
) -> list[Chain]:
    """Draw up to ``count`` unique chains. Stops early if the space is exhausted."""
    pool: list[Chain] = []
    seen: set[tuple] = set()
    attempts = 0
    stall = 0
    limit = max(count * 40, 100_000)
    while len(pool) < count and attempts < limit and stall < 20_000:
        attempts += 1
        chain = _sample_chain(rng, k, template_numbers)
        if chain is None:
            stall += 1
            continue
        key = chain.key()
        if key in banned_keys or key in seen:
            stall += 1
            continue
        stall = 0
        seen.add(key)
        pool.append(chain)
    return pool


def _decile(value: int) -> int:
    return min(value // 10, 9)


def select_balanced(pool: list[Chain], need: int) -> list[Chain]:
    """Greedy selection. Ties keep the earlier item in key order."""
    remaining = sorted(pool, key=lambda chain: chain.key())
    selected: list[Chain] = []
    decile = [0] * 10
    op_counts = {op: 0 for op in OPS}
    n_inter = 0
    n_ops = 0

    def deviation(extra_d: list[int], extra_o: dict[str, int], ni: int, no: int) -> float:
        devs: list[float] = []
        if ni:
            for i in range(10):
                devs.append(abs((decile[i] + extra_d[i]) / ni - 0.1))
        if no:
            for op in OPS:
                devs.append(abs((op_counts[op] + extra_o[op]) / no - 1 / 3))
        return max(devs) if devs else 0.0

    for _ in range(need):
        best_i = 0
        best_dev = None
        for i, chain in enumerate(remaining):
            extra_d = [0] * 10
            extra_o = {op: 0 for op in OPS}
            for value in chain.intermediates:
                extra_d[_decile(value)] += 1
            for op in chain.ops:
                extra_o[op] += 1
            dev = deviation(
                extra_d,
                extra_o,
                n_inter + len(chain.intermediates),
                n_ops + len(chain.ops),
            )
            if best_dev is None or dev < best_dev:
                best_dev = dev
                best_i = i
        chain = remaining.pop(best_i)
        selected.append(chain)
        for value in chain.intermediates:
            decile[_decile(value)] += 1
            n_inter += 1
        for op in chain.ops:
            op_counts[op] += 1
            n_ops += 1
    return selected


def balance_report(chains: list[Chain]) -> dict:
    decile = [0] * 10
    op_counts = {op: 0 for op in OPS}
    for chain in chains:
        for value in chain.intermediates:
            decile[_decile(value)] += 1
        for op in chain.ops:
            op_counts[op] += 1
    n_inter = sum(decile)
    n_ops = sum(op_counts.values())
    decile_share = [c / n_inter if n_inter else 0.0 for c in decile]
    op_share = {op: op_counts[op] / n_ops if n_ops else 0.0 for op in OPS}
    gate = None
    if chains and chains[0].k >= 2 and n_inter:
        gate = all(0.06 <= s <= 0.14 for s in decile_share) and all(
            0.25 <= op_share[op] <= 0.42 for op in OPS
        )
    return {"decile_share": decile_share, "op_share": op_share, "gate": gate}


def predecessors(op: str, target: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    if op == "+":
        for prev in range(0, target):
            n = target - prev
            if 1 <= n <= 99 - prev:
                found.append((prev, n))
    elif op == "-":
        for n in range(1, 100 - target):
            prev = target + n
            if prev <= 99 and 1 <= n <= prev:
                found.append((prev, n))
    elif op == "*":
        for n in range(2, 10):
            if target % n == 0:
                prev = target // n
                if 0 <= prev <= 99 and prev * n == target:
                    found.append((prev, n))
    return found


def random_prefix(ops: tuple[str, ...], target: int, rng: np.random.Generator) -> tuple[int, tuple[int, ...]] | None:
    operands_rev: list[int] = []
    value = target
    for index, op in enumerate(reversed(ops)):
        choices = predecessors(op, value)
        if index == len(ops) - 1:
            choices = [(prev, n) for prev, n in choices if 1 <= prev <= 99]
        if not choices:
            return None
        prev, n = choices[int(rng.integers(0, len(choices)))]
        operands_rev.append(n)
        value = prev
    return value, tuple(reversed(operands_rev))


def continue_from(start: int, ops: tuple[str, ...], operands: tuple[int, ...]) -> list[int] | None:
    values = [start]
    v = start
    for op, n in zip(ops, operands, strict=True):
        moves = legal_moves(v)
        if (op, n) not in moves:
            return None
        v = apply_op(v, op, n)
        if not 0 <= v <= 99:
            return None
        values.append(v)
    return values


def valid_swap_values(chain: Chain, step: int, template_numbers: set[int]) -> list[tuple[int, int]]:
    """step is 1-based. Returns (b, y_b) pairs."""
    a = chain.values[step]
    suffix_ops = chain.ops[step:]
    suffix_operands = chain.operands[step:]
    prompt_numbers = {chain.x0, *chain.operands}
    earlier = set(chain.values[1:step])
    found = []
    for b in range(100):
        if b == a or b in prompt_numbers or b in earlier or b in template_numbers:
            continue
        walked = continue_from(b, suffix_ops, suffix_operands)
        if walked is None:
            continue
        y_b = walked[-1]
        later = walked[1:-1]
        if y_b == chain.answer or y_b in template_numbers or y_b in prompt_numbers:
            continue
        if y_b in earlier or y_b == b or any(v in prompt_numbers or v in template_numbers or v == y_b for v in later):
            continue
        if len(set(later)) != len(later) or b in later:
            continue
        found.append((b, y_b))
    return found


def prompts_aligned(tok, base: str, variant: str) -> bool:
    base_ids = tok.encode(base, add_special_tokens=False)
    var_ids = tok.encode(variant, add_special_tokens=False)
    if len(base_ids) != len(var_ids):
        return False
    for left, right in zip(base_ids, var_ids, strict=True):
        if left == right:
            continue
        left_text = tok.decode([left]).strip()
        right_text = tok.decode([right]).strip()
        if not (left_text.isdigit() and right_text.isdigit()):
            return False
    return True


def template_numbers_from_render(tok, rendered: str, problem: str) -> set[int]:
    start = rendered.index(problem)
    end = start + len(problem)
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    numbers: set[int] = set()
    for token_id, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True):
        if right <= start or left >= end:
            text = tok.decode([token_id]).strip()
            if text.isdigit() and int(text) <= 99:
                numbers.add(int(text))
    return numbers


def compute_template_numbers(tok) -> set[int]:
    problem = "PROBLEM_SENTINEL"
    numbers: set[int] = set()
    for user in (
        f"{problem}\nAnswer with only the final number and nothing else.",
        problem,
    ):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        numbers |= template_numbers_from_render(tok, rendered, problem)
    return numbers


def build_variants(chain: Chain, fmt: str, tok, template_numbers: set[int], mul_symbol: str = MUL_SYMBOL) -> dict:
    """Matched variants for intermediates. k=1 returns empty lists."""
    empty: dict = {
        "swap_b": [],
        "swap_yb": [],
        "patch_src_text": [],
        "patch_ctrl_text": [],
        "null_values": [],
    }
    if chain.k == 1:
        return empty
    base = render(chain, fmt, mul_symbol)
    item_id = chain.item_id(fmt)
    for step in range(1, chain.k):
        rng = np.random.default_rng(hash64(item_id, step))
        pairs = valid_swap_values(chain, step, template_numbers)
        if pairs:
            b, y_b = pairs[int(rng.integers(0, len(pairs)))]
        else:
            b, y_b = None, None
        excluded = {chain.x0, *chain.operands, *chain.intermediates, chain.answer, *template_numbers}
        null_pool = [n for n in range(100) if n not in excluded]
        null_value = int(null_pool[int(rng.integers(0, len(null_pool)))]) if null_pool else None
        src_text = _search_patch(chain, step, b, rng, tok, base, fmt, template_numbers, mul_symbol, must_differ=False)
        ctrl_text = _search_patch(
            chain, step, chain.values[step], rng, tok, base, fmt, template_numbers, mul_symbol, must_differ=True
        )
        empty["swap_b"].append(b)
        empty["swap_yb"].append(y_b)
        empty["null_values"].append(null_value)
        empty["patch_src_text"].append(src_text)
        empty["patch_ctrl_text"].append(ctrl_text)
    return empty


def _search_patch(chain, step, target, rng, tok, base, fmt, template_numbers, mul_symbol, must_differ: bool):
    if target is None:
        return None
    original_prefix = (chain.x0, chain.operands[:step])
    for _ in range(200):
        prefix = random_prefix(chain.ops[:step], target, rng)
        if prefix is None:
            continue
        x0, early = prefix
        if must_differ and (x0, early) == original_prefix:
            continue
        if not must_differ and (x0, early) == original_prefix:
            continue
        operands = early + chain.operands[step:]
        variant = Chain(x0, chain.ops, operands)
        if variant.values[step] != target or not constraints_ok(variant, template_numbers):
            continue
        text = render(variant, fmt, mul_symbol)
        if prompts_aligned(tok, base, text):
            return text
    return None


def generate_items(template_numbers: set[int], tok, mul_symbol: str = MUL_SYMBOL) -> list[dict]:
    counts = {"dev": 50, "pilot": 200, "main": 300}
    seeds = {"dev": 1, "pilot": 2, "main": 3}
    banned: set[tuple] = set()
    rows: list[dict] = []
    for split, seed in seeds.items():
        for fmt in ("nested", "word"):
            for k in range(1, 7):
                need = counts[split]
                rng = np.random.default_rng(seed * 1000 + k)
                pool = draw_pool(rng, k, 50 * need, template_numbers, banned)
                if len(pool) < need:
                    raise RuntimeError(f"{split} {fmt} k={k}: pool {len(pool)} < needed {need}")
                if len(pool) < 50 * need:
                    print(f"{split} {fmt} k={k}: pool {len(pool)} < 50x {need}", flush=True)
                chosen = select_balanced(pool, need)
                for chain in chosen:
                    banned.add(chain.key())
                    variants = build_variants(chain, fmt, tok, template_numbers, mul_symbol)
                    rows.append(
                        {
                            "item_id": chain.item_id(fmt),
                            "format": fmt,
                            "split": split,
                            "k": k,
                            "x0": chain.x0,
                            "ops": list(chain.ops),
                            "operands": list(chain.operands),
                            "values": list(chain.values),
                            "answer": chain.answer,
                            "problem_text": render(chain, fmt, mul_symbol),
                            "intermediates": list(chain.intermediates),
                            **variants,
                        }
                    )
    return rows
