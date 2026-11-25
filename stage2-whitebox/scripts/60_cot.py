"""CoT generation. Refuses unless on AC.

Torch is the backend that runs here. MLX is selected only after
``logs/preflight/backend_ok`` exists; otherwise torch is used and that fact
is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.cot_gen import answer_after_think  # noqa: E402
from s2.models import load  # noqa: E402
from s2.runio import require_ac_power  # noqa: E402
from s2.tasks import render_prompt  # noqa: E402


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    ordered, indices = torch.sort(probs, descending=True)
    keep = ordered.cumsum(0) <= top_p
    keep[0] = True
    chosen = torch.multinomial(ordered * keep, 1)
    return int(indices[chosen])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="main")
    parser.add_argument("--format", default="nested")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    require_ac_power()
    if not (ROOT / "logs" / "preflight" / "backend_ok").is_file():
        print("MLX backend check missing; using torch")
    key = f"{args.model}.{args.ckpt}"
    model = load(key, device=args.device)
    items = pd.read_parquet(ROOT / "data" / f"{args.format}_{args.split}.parquet").head(args.limit)
    out = ROOT / "results" / key / "cot" / "cot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            if line:
                done.add(json.loads(line)["item_id"])
    with out.open("a") as handle:
        for item in items.itertuples(index=False):
            if item.item_id in done:
                continue
            _text, ids = render_prompt(model.tok, item.problem_text, "cot")
            input_ids = torch.tensor([ids], device=args.device)
            generated = []
            with torch.no_grad():
                out_fwd = model.hf(input_ids=input_ids, use_cache=True)
                cache = out_fwd.past_key_values
                nxt = sample(out_fwd.logits[0, -1], 0.6, 0.95)
                generated.append(nxt)
                for _ in range(args.max_new_tokens - 1):
                    out_fwd = model.hf(input_ids=torch.tensor([[nxt]], device=args.device), past_key_values=cache, use_cache=True)
                    cache = out_fwd.past_key_values
                    nxt = sample(out_fwd.logits[0, -1], 0.6, 0.95)
                    generated.append(nxt)
                    if nxt in (model.tok.eos_token_id,):
                        break
            text = model.tok.decode(generated)
            record = {
                "item_id": item.item_id,
                "k": int(item.k),
                "answer": int(item.answer),
                "text": text,
                "ids": generated,
                "finish_reason": "eos" if generated[-1] == model.tok.eos_token_id else "length",
                "parsed": answer_after_think(text),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(item.item_id, record["finish_reason"], flush=True)


if __name__ == "__main__":
    main()
