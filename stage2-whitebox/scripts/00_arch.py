"""Gate 0.2. Load Olmo-3-7B-Think-DPO by SHA and record the block structure.

Long job: refuses unless the laptop is on AC power.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import require_ac_power, thermal_note, write_manifest  # noqa: E402


def module_tree(module: torch.nn.Module, prefix: str = "", depth: int = 0, limit: int = 3) -> list[str]:
    lines = []
    if depth > limit:
        return lines
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        extra = ""
        if isinstance(child, torch.nn.Linear):
            extra = f" Linear in={child.in_features} out={child.out_features} bias={child.bias is not None}"
        elif hasattr(child, "weight") and getattr(child, "weight", None) is not None:
            extra = f" weight={tuple(child.weight.shape)} bias={getattr(child, 'bias', None) is not None}"
        lines.append(f"{'  ' * depth}- `{path}` {child.__class__.__name__}{extra}")
        lines.extend(module_tree(child, path, depth + 1, limit))
    return lines


def has_bias(module: torch.nn.Module) -> bool:
    bias = getattr(module, "bias", None)
    return bias is not None


def main() -> None:
    require_ac_power()
    before = thermal_note()
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    repo, revision = entry["repo"], entry["revision"]
    print(f"loading {repo}@{revision}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("mps")
    cfg = model.config
    block = model.model.layers[0]
    final_norm = model.model.norm
    lm_head = model.lm_head
    layer_types = list(getattr(cfg, "layer_types", []) or [])
    lines = [
        "# Olmo-3-7B-Think-DPO architecture",
        "",
        f"- repo `{repo}` revision `{revision}`",
        f"- dtype {next(model.parameters()).dtype} device {next(model.parameters()).device}",
        f"- layers {cfg.num_hidden_layers} (expect 32)",
        f"- d_model {cfg.hidden_size} (expect 4096)",
        f"- heads {cfg.num_attention_heads} kv_heads {cfg.num_key_value_heads} (expect 32)",
        f"- intermediate {cfg.intermediate_size} (expect 11008)",
        f"- vocab {cfg.vocab_size} (expect 100278)",
        f"- tie_word_embeddings {cfg.tie_word_embeddings} (expect false)",
        f"- sliding_window {getattr(cfg, 'sliding_window', None)} (expect 4096)",
        f"- attn_implementation {getattr(cfg, '_attn_implementation', None)}",
        f"- layer_types {layer_types}",
        f"- final_norm class `{final_norm.__class__.__name__}` bias={has_bias(final_norm)}",
        f"- lm_head class `{lm_head.__class__.__name__}` bias={has_bias(lm_head)}",
        "",
        "## Block 0 module tree",
        "",
        *module_tree(block),
        "",
        "## Norm modules",
        "",
    ]
    for name, module in model.named_modules():
        if "norm" in module.__class__.__name__.lower() or name.endswith("norm"):
            lines.append(f"- `{name}` {module.__class__.__name__} bias={has_bias(module)}")
    problems = []
    if cfg.num_hidden_layers != 32:
        problems.append(f"layers {cfg.num_hidden_layers}")
    if cfg.hidden_size != 4096:
        problems.append(f"d_model {cfg.hidden_size}")
    if cfg.num_key_value_heads != 32:
        problems.append(f"kv_heads {cfg.num_key_value_heads}")
    if has_bias(final_norm):
        problems.append("final_norm has a bias")
    if not isinstance(lm_head, torch.nn.Linear) or has_bias(lm_head):
        problems.append(f"lm_head is {type(lm_head).__name__} bias={has_bias(lm_head)}")
    if problems:
        lines.extend(["", "## Mismatches", ""])
        lines.extend(f"- {item}" for item in problems)
    out = ROOT / "logs" / "preflight" / "olmo3_7b_arch.md"
    out.write_text("\n".join(lines) + "\n")
    after = thermal_note()
    write_manifest(
        ROOT / "logs" / "preflight",
        script="00_arch.py",
        model_id=repo,
        revision=revision,
        thermal_note={"before": before, "after": after},
    )
    print(f"wrote {out}")
    if problems:
        raise SystemExit("architecture mismatches: " + "; ".join(problems))


if __name__ == "__main__":
    main()
