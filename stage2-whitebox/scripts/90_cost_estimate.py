"""32B cost estimate. Refuses to print dollars without --usd-per-gpu-hour."""

from __future__ import annotations

import argparse
from pathlib import Path

# Published bf16 peaks used only as a documented ratio, not a measurement.
H200_BF16_TFLOPS = 990
M5_MEASURED_TFLOPS = 30.0  # Gate 0.3, 4096 matmul, torch MPS
PARAMS_32_OVER_7 = 32 / 7
WIDTH_32_OVER_7 = 5120 / 4096
H200_TOK_PER_S = 1050


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd-per-gpu-hour", type=float, default=None)
    parser.add_argument("--fit-hours-7b", type=float, default=None)
    parser.add_argument("--generated-tokens-7b", type=float, default=None)
    args = parser.parse_args()
    ratio = (H200_BF16_TFLOPS / M5_MEASURED_TFLOPS)
    fit_7b = args.fit_hours_7b
    fit_32 = None if fit_7b is None else fit_7b * PARAMS_32_OVER_7 * WIDTH_32_OVER_7 / ratio
    gen_hours = None if args.generated_tokens_7b is None else args.generated_tokens_7b / H200_TOK_PER_S / 3600
    lines = [
        "# 32B cost estimate",
        "",
        f"Throughput ratio H200/M5 = {H200_BF16_TFLOPS}/{M5_MEASURED_TFLOPS} = {ratio:.1f}.",
        "H200 bf16 TFLOPS is the published dense peak, not a measurement on this code.",
        f"Fit scale = (32/7) * (5120/4096) / {ratio:.1f}.",
        f"Generation uses {H200_TOK_PER_S} tok/s.",
        "",
        f"- 7B fit hours: {fit_7b}",
        f"- 32B fit hours (one checkpoint): {fit_32}",
        f"- 32B generation hours: {gen_hours}",
        "",
    ]
    if args.usd_per_gpu_hour is None:
        lines.append("Dollars omitted: pass --usd-per-gpu-hour.")
    else:
        hours = (fit_32 or 0) * 4 + (gen_hours or 0) * 4
        lines.append(f"At ${args.usd_per_gpu_hour}/GPU-hour, 4 checkpoints, fit+generation ≈ ${hours * args.usd_per_gpu_hour:.0f}.")
        lines.append("Readout, swap and patching scale like the fit and are not added until those 7B times exist.")
    out = Path(__file__).resolve().parents[1] / "cloud" / "COST_ESTIMATE.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
