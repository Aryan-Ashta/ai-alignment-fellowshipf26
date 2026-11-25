"""Gate 0.5. Peak MPS driver memory for load, fit-sized backward, and a 16k forward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import require_ac_power, thermal_note, write_manifest  # noqa: E402


def peak() -> int:
    torch.mps.synchronize()
    return int(torch.mps.driver_allocated_memory())


def main() -> None:
    require_ac_power()
    before = thermal_note()
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    repo, revision = entry["repo"], entry["revision"]
    torch.mps.empty_cache()
    base = peak()
    print(f"loading {repo}@{revision}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to("mps")
    model.eval()
    model.requires_grad_(False)
    load_peak = peak() - base

    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (8, 128), device="mps")
    embeds = model.model.embed_tokens(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds).logits
    logits.float().sum().backward()
    torch.mps.synchronize()
    backward_peak = peak() - base
    del embeds, logits
    torch.mps.empty_cache()

    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    long_ids = torch.randint(0, vocab, (1, 16384), device="mps")
    with torch.no_grad():
        _ = model(input_ids=long_ids)
    torch.mps.synchronize()
    long_peak = peak() - base
    del model, long_ids, tok
    lines = [
        "# Gate 0.5 memory",
        "",
        "Peaks are `torch.mps.driver_allocated_memory()` minus the reading before load.",
        "",
        f"- model load: {load_peak / 1e9:.2f} GB",
        f"- forward+backward batch 8×128: {backward_peak / 1e9:.2f} GB",
        f"- forward 1×16384: {long_peak / 1e9:.2f} GB",
        "",
    ]
    out = ROOT / "logs" / "preflight" / "memory.md"
    out.write_text("\n".join(lines))
    write_manifest(
        ROOT / "logs" / "preflight",
        name="manifest_00_memory.json",
        script="00_memory.py",
        model_id=repo,
        revision=revision,
        peak_memory={
            "load": load_peak,
            "forward_backward_8x128": backward_peak,
            "forward_16k": long_peak,
        },
        thermal_note={"before": before, "after": thermal_note()},
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
