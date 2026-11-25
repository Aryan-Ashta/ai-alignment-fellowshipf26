"""Gate 0.3. Matmul throughput and one 7B forward+backward.

Decision recorded here: the fitter stays torch-MPS. MLX is for CoT generation only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import torch
import yaml
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import require_ac_power, thermal_note, write_manifest  # noqa: E402


def torch_matmul(m: int, k: int, n: int) -> tuple[float, float]:
    a = torch.randn(m, k, device="mps", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="mps", dtype=torch.bfloat16)
    for _ in range(3):
        _ = a @ b
        torch.mps.synchronize()
    best = None
    for _ in range(5):
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = a @ b
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    flops = 2 * m * k * n
    return best, flops / best / 1e12


def mlx_matmul(m: int, k: int, n: int) -> tuple[float, float]:
    a = mx.random.normal(shape=(m, k)).astype(mx.bfloat16)
    b = mx.random.normal(shape=(k, n)).astype(mx.bfloat16)
    for _ in range(3):
        mx.eval(a @ b)
    best = None
    for _ in range(5):
        start = time.perf_counter()
        mx.eval(a @ b)
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    flops = 2 * m * k * n
    return best, flops / best / 1e12


def forward_backward(model) -> float:
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (8, 128), device="mps")
    for _ in range(1):
        embeds = model.model.embed_tokens(ids).detach().requires_grad_(True)
        logits = model(inputs_embeds=embeds).logits
        logits.float().sum().backward()
        torch.mps.synchronize()
    torch.mps.synchronize()
    start = time.perf_counter()
    embeds = model.model.embed_tokens(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds).logits
    logits.float().sum().backward()
    torch.mps.synchronize()
    return time.perf_counter() - start


def main() -> None:
    require_ac_power()
    before = thermal_note()
    lines = ["# Gate 0.3 throughput", "", "Best of 5 after 3 warm-up calls. bf16.", ""]
    rows = []
    for m, k, n in ((4096, 4096, 4096), (8192, 4096, 4096)):
        t_s, t_tf = torch_matmul(m, k, n)
        x_s, x_tf = mlx_matmul(m, k, n)
        rows.append((f"{m}x{k}x{n}", t_s, t_tf, x_s, x_tf, x_tf / t_tf))
        lines.append(
            f"- {m}×{k}×{n}: torch {t_tf:.2f} TFLOPS ({t_s*1e3:.1f} ms), "
            f"MLX {x_tf:.2f} TFLOPS ({x_s*1e3:.1f} ms), MLX/torch {x_tf / t_tf:.2f}"
        )
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    print(f"loading {entry['repo']}@{entry['revision']}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        entry["repo"],
        revision=entry["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to("mps")
    model.eval()
    model.requires_grad_(False)
    fb = forward_backward(model)
    lines.append(f"- 7B forward+backward batch 8×128, grads w.r.t. inputs only: {fb:.2f} s")
    mlx_faster = all(row[5] >= 2 for row in rows)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The fitter stays torch-MPS. MLX is used only for CoT generation.",
            f"MLX matmul ≥ 2× torch on every shape: {mlx_faster}.",
            "A fitter port to MLX is not in this plan. Gate 3.4 still stops if a checkpoint fit projects past 12 h.",
            "",
        ]
    )
    out = ROOT / "logs" / "preflight" / "bench.md"
    out.write_text("\n".join(lines))
    write_manifest(
        ROOT / "logs" / "preflight",
        name="manifest_00_bench.json",
        script="00_bench.py",
        model_id=entry["repo"],
        revision=entry["revision"],
        thermal_note={"before": before, "after": thermal_note()},
        wall_time_s=fb,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
