"""Gate 0.1. Number-token identity and in-context tokenization.

Blocking. Stops if any of 0–99 merges with a leading space into one token.
Downloads tokenizer files only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import write_manifest  # noqa: E402

MUL_SYMBOLS = ("*", "×", "x", "times")


def load_models() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())


def tokenizer_jobs(models: dict) -> list[tuple[str, str, dict]]:
    jobs = [(f"olmo3_7b.{slot}", "olmo3", entry) for slot, entry in models["olmo3_7b"].items()]
    jobs.append(("olmo3_32b.dpo", "olmo3", models["olmo3_32b"]["dpo"]))
    jobs.append(("codi", "codi", models["codi"]))
    return jobs


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def describe(tok, text: str) -> str:
    pieces = encode(tok, text)
    decoded = [tok.decode([i]) for i in pieces]
    return f"{text!r} -> {pieces} {decoded}"


def number_ids(tok) -> dict[str, int]:
    ids: dict[str, int] = {}
    for n in range(100):
        text = str(n)
        pieces = encode(tok, text)
        if len(pieces) != 1:
            raise SystemExit(f"{text!r} encodes to {len(pieces)} tokens {pieces}")
        ids[text] = pieces[0]
    return ids


def space_merged_values(tok) -> list[int]:
    return [n for n in range(100) if len(encode(tok, f" {n}")) == 1]


def context_lines(tok) -> list[str]:
    rows = [
        describe(tok, " 42"),
        describe(tok, "= 42"),
        describe(tok, "(17 + 25)"),
        describe(tok, "x=42"),
        describe(tok, "\n42"),
        describe(tok, "</think>42"),
        describe(tok, "</think>\n\n42"),
        describe(tok, " * "),
        describe(tok, " × "),
    ]
    for symbol in MUL_SYMBOLS:
        rows.append(describe(tok, f"(17 + 25) {symbol} 3 ="))
        rows.append(describe(tok, f"Multiply by {symbol}." if symbol == "times" else f"Multiply by 3."))
    # "Multiply by 3." is required once; the loop repeats it. Keep one explicit row.
    rows.append(describe(tok, "Multiply by 3."))
    return rows


def main() -> None:
    models = load_models()
    out_dir = ROOT / "logs" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Tokenizer preflight (Gate 0.1)", ""]
    family_ids: dict[str, dict[str, int]] = {}
    family_meta: dict[str, dict] = {}
    olmo_ids: list[tuple[str, dict[str, int]]] = []
    merges: list[tuple[str, list[int]]] = []

    for name, family, entry in tokenizer_jobs(models):
        repo, revision = entry["repo"], entry["revision"]
        print(f"loading {name} {repo}@{revision}", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        ids = number_ids(tok)
        merged = space_merged_values(tok)
        if family not in family_ids:
            family_ids[family] = ids
            family_meta[family] = {
                "repo": repo,
                "revision": revision,
                "number_token_ids": ids,
                "space_number_merge": bool(merged),
                "space_merged_values": merged,
            }
        if family == "olmo3":
            olmo_ids.append((name, ids))
        if merged:
            merges.append((name, merged))
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- repo `{repo}` revision `{revision}`")
        lines.append("- 100 single-token ids: yes")
        lines.append(f"- id('0')={ids['0']} id('42')={ids['42']} id('99')={ids['99']}")
        lines.append(f"- space_number_merge: {bool(merged)}")
        if merged:
            lines.append(f"- merged values: {merged}")
        lines.append("- in-context:")
        for row in context_lines(tok):
            lines.append(f"  - {row}")
        lines.append("")

    ref_name, ref_ids = olmo_ids[0]
    mismatches = [name for name, ids in olmo_ids[1:] if ids != ref_ids]
    if mismatches:
        lines.append(f"FAIL: number ids differ from {ref_name}: {mismatches}")
        (out_dir / "tokenizer.md").write_text("\n".join(lines) + "\n")
        raise SystemExit(f"number ids differ from {ref_name}: {mismatches}")
    lines.append(f"Olmo number ids identical across {len(olmo_ids)} tokenizers (reference `{ref_name}`).")
    codi_match = family_ids["codi"] == ref_ids
    lines.append(
        "CODI number ids " + ("match Olmo." if codi_match else "differ from Olmo (separate tokenizer family).")
    )
    lines.append("")

    family_merges = {
        "olmo3": sorted({n for name, vals in merges if not name.startswith("codi") for n in vals}),
        "codi": sorted({n for name, vals in merges if name.startswith("codi") for n in vals}),
    }
    for family, record in family_meta.items():
        record["space_merged_values"] = family_merges[family]
        record["space_number_merge"] = bool(family_merges[family])
        path = out_dir / f"number_token_ids_{family}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        lines.append(f"Wrote `{path.name}` space_number_merge={record['space_number_merge']}.")

    (out_dir / "tokenizer.md").write_text("\n".join(lines) + "\n")
    write_manifest(out_dir, script="00_tokenizer.py", model_id="tokenizers")
    print(f"wrote {out_dir / 'tokenizer.md'}")
    if merges:
        raise SystemExit(
            "space/number merge detected; readout definition must be updated: "
            + ", ".join(f"{name}:{vals}" for name, vals in merges)
        )


if __name__ == "__main__":
    main()
