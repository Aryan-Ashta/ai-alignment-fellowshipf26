#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/persistent/hf}"
uv run python scripts/run_checkpoint.py --model olmo3_32b --ckpt mid --device cuda --attn sdpa --results-root /persistent/results --split main --steps hooks,router,fit,readout,nocot,swap,patch,cot
