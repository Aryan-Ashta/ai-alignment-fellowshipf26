"""Long WikiText records for the fit. Streaming, so the full corpus is not downloaded."""

from __future__ import annotations


def wikitext_long(n: int, min_chars: int = 600) -> list[str]:
    from datasets import load_dataset

    texts = []
    for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True):
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        texts.append(text)
        if len(texts) == n:
            return texts
    raise RuntimeError(f"WikiText yielded {len(texts)} long records, need {n}")
