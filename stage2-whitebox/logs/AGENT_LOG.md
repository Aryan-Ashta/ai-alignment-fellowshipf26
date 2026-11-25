# Stage 2 agent log

## 2026-11-25

### P0.1 layout

- Did: created `stage2-whitebox/` in the existing repo (not a new git root). Tree matches SPEC §2.1 plus `src/s2/codi.py` / `src/s2/runio.py` / `cloud/` (added in later commits as those modules land). `.gitignore` excludes `scratch/`, `third_party/`, `.venv/`, `**/*.bin`, `**/*.pt`, `lenses/**/*.safetensors`, `results/**/raw/`, `__pycache__/`.
- Passed: directory tree exists; spec file `stage2_whitebox_agent_plan.md` left untracked and unedited.
- Failed: nothing.
- Decided: project lives in this repo, not `~/research/stage2-whitebox/`. The approved plan overrides SPEC §2.1's default location. `stage2_whitebox_agent_plan.md` is never committed from this work.

### P0.2 third-party pins

- Did: wrote `scripts/setup_third_party.sh` and `configs/third_party.yaml` with SHAs `jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e`, `jpp_lens@d7efc0de372e20589e899188c0cfe8434acc63ec`, `codi@f0bf2fa0f5a54c749395a5d095bc406ad18b8efc`.
- Passed: all three clones detached at the pinned SHAs; working trees clean.
- Failed: nothing.
- Decided: clones stay gitignored; only the pin file and script are committed.
