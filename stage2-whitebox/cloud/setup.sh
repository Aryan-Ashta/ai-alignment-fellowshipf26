#!/usr/bin/env bash
# Prepare a 32B box at the run commit. Does not launch a fit.
# Torch comes from the cu128 index at the same version as pyproject.toml.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

REPO_URL="${REPO_URL:-https://github.com/Aryan-Ashta/ai-alignment-fellowshipf26.git}"
RUN_COMMIT="${RUN_COMMIT:?set RUN_COMMIT to the freeze sha}"
ROOT="${ROOT:-$HOME/stage2-whitebox-run}"

git clone "$REPO_URL" "$ROOT"
git -C "$ROOT" checkout --detach "$RUN_COMMIT"
cd "$ROOT/stage2-whitebox"
bash scripts/setup_third_party.sh

TORCH_VERSION="$(uv run python -c 'import tomllib; print(next(l.split("==")[1].strip("\",") for l in open("pyproject.toml") if l.strip().startswith("\"torch==")))')"
uv sync --group cloud || uv sync
export HF_HOME="${HF_HOME:-/persistent/hf}"
echo "HF_HOME=$HF_HOME"
echo "Expected torch==$TORCH_VERSION from the cu128 index. If that wheel is missing, stop and ask."
