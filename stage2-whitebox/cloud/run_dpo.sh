#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../stage2-whitebox"
export HF_HOME="${HF_HOME:-/persistent/hf}"
uv run python scripts/run_checkpoint.py --model olmo3_32b --ckpt dpo --device cuda --attn sdpa --results-root /persistent/results --split main
