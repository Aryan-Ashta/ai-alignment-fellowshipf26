"""Separate MPS error from bf16-vs-fp32 error for Gate 0.4.

The fp32 reference is already in scratch/attn_ref. This compares one MPS bf16
sdpa forward to that reference and to a CPU bf16 sdpa forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import require_ac_power  # noqa: E402


def load(repo, revision, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def last_block(model, ids):
    captured = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[-1].register_forward_hook(hook)
    try:
        logits = model(input_ids=ids).logits.detach()
    finally:
        handle.remove()
    return captured["h"].cpu(), logits.cpu()


def report(name, ref_h, ref_arg, cand_h, cand_arg):
    cos = F.cosine_similarity(ref_h[0].float(), cand_h[0].float(), dim=-1)
    agree = float((ref_arg == cand_arg).float().mean())
    print(
        f"{name}: min_cos={float(cos.min()):.5f} p50={float(cos.median()):.5f} "
        f"frac<0.995={float((cos < 0.995).float().mean()):.4f} argmax={agree:.4f}",
        flush=True,
    )


def main() -> None:
    require_ac_power()
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    repo, revision = entry["repo"], entry["revision"]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    # Rebuild the same WikiText prefix the gate used: first 16384 tokens of the stream.
    from datasets import load_dataset

    parts = []
    for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True):
        if row["text"].strip():
            parts.append(row["text"].strip())
        ids = tok.encode("\n".join(parts), add_special_tokens=False)
        if len(ids) >= 8192:
            break
    ids = torch.tensor(ids[:8192], dtype=torch.long)[None, :]
    for n in (4096, 8192):
        ref = torch.load(ROOT / "scratch" / "attn_ref" / f"cpu_fp32_{n}.pt", weights_only=True)
        print(f"--- {n} ---", flush=True)
        mps = load(repo, revision, torch.bfloat16, "mps")
        h, logits = last_block(mps, ids[:, :n].to("mps"))
        torch.mps.synchronize()
        report("mps-bf16 vs cpu-fp32", ref["hidden"], ref["logits"], h, logits.argmax(-1))
        del mps
        cpu = load(repo, revision, torch.bfloat16, "cpu")
        ch, cl = last_block(cpu, ids[:, :n])
        report("cpu-bf16 vs cpu-fp32", ref["hidden"], ref["logits"], ch, cl.argmax(-1))
        report("mps-bf16 vs cpu-bf16", ch, cl.argmax(-1), h, logits.argmax(-1))
        del cpu


if __name__ == "__main__":
    main()
