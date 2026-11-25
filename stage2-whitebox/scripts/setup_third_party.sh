#!/usr/bin/env bash
# Clone pinned reference repos into third_party/ and detach at the recorded SHAs.
# Never edits the trees. Idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
mkdir -p "$DEST"

clone_pin() {
  local url="$1" name="$2" sha="$3"
  local dir="$DEST/$name"
  if [[ ! -d "$dir/.git" ]]; then
    rm -rf "$dir"
    git clone "$url" "$dir"
  fi
  git -C "$dir" fetch origin "$sha"
  git -C "$dir" checkout --detach "$sha"
  local got
  got="$(git -C "$dir" rev-parse HEAD)"
  if [[ "$got" != "$sha" ]]; then
    echo "SHA mismatch for $name: got $got want $sha" >&2
    exit 1
  fi
  echo "$name $got"
}

clone_pin https://github.com/anthropics/jacobian-lens.git jacobian-lens 581d398613e5602a5af361e1c34d3a92ea82ba8e
clone_pin https://github.com/safety-research/jpp_lens.git jpp_lens d7efc0de372e20589e899188c0cfe8434acc63ec
clone_pin https://github.com/cywinski/codi.git codi f0bf2fa0f5a54c749395a5d095bc406ad18b8efc
