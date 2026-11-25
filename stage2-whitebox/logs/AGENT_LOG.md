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

### P0.3 Python env

- Did: wrote `pyproject.toml` (`requires-python ==3.12.*`) matching jpp_lens pins, plus `torch==2.13.0` from default PyPI. `uv lock` resolved the unpinned directs; those versions were written back as `==` and the lock regenerated. `jpp-lens` is a non-editable path source. anthropics `jlens` is not installed (module name collides with jpp's vendored `jlens`). No scikit-learn, no peft. `HF_HOME` left at the default.
- Passed: `uv sync` installed 76 packages. `torch 2.13.0`, `torch.backends.mps.is_available()` is True, `transformers 5.14.1`, `import s2` and `import workspace_lens` succeed. mlx-lm resolved to 0.32.2 (≥ 0.28.2).
- Failed: nothing.
- Decided: torch 2.13.0 was installable, so no fallback. accelerate resolved to 1.16.0 (jpp's extra pins 1.12.0; we keep the PyPI resolution next to the MPS wheel). Cloud cu128 index is deferred to P10 so the default source stays PyPI and MPS is not replaced by a CPU wheel.

### P0.4 checkpoint resolution

- Did: `scripts/00_resolve_checkpoints.py` resolved every SPEC §2.4 default plus CODI and `unsloth/Llama-3.2-1B-Instruct` via `HfApi.list_repo_refs` and `model_info(files_metadata=True)`. Wrote `configs/models.yaml` and `logs/preflight/checkpoint_identity.md`.
- Passed: live SHAs match the 2026-11-25 facts. 7B `main` weight sha256s differ from `step_1375`. 32B `Olmo-3.1@step_0750` ≡ `Olmo-3@step_750`. 32B `Olmo-3.1@main` ≡ `step_2300`. `codi_base` `model.safetensors` is 2471645608 bytes at `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. `tests/test_runio.py` 4 passed. `tests/test_checkpoints.py` asserts revisions are 40-hex SHAs and no `from_pretrained` call passes a branch literal.
- Failed: nothing.
- Decided: 7B final `rl_step` is 1375 with `label: main` (bytes differ from `step_1375`). 32B final `rl_step` is 2300 with `label: main` because that revision is byte-identical to `step_2300`. `shard_sha256` stores file, sha256, and size so both safetensors and bin shards stay auditable.

### P0.6 Gate 0.1 tokenizer

- Did: ran `scripts/00_tokenizer.py` on the four 7B revisions, 32B DPO, and CODI, all by revision SHA.
- Passed: every `"0"`…`"99"` is one token. Ids are identical across the five Olmo tokenizers and also match CODI. `space_number_merge` is false; `" 42"` is `[space, 42]`, not one token. No stop.
- Failed: nothing.
- Decided: `*` and `×` both absorb the leading space into one token and leave the surrounding ids unchanged (`(17 + 25) * 3 =` vs `×` differ only in the operator id). Plan says use `*` unless that holds, so `tasks.yaml: mul_symbol` will be `×`. `</think>` is not one token (`</` + `think` + `>`); prefill calibration will append the string and measure the first generated token, not assume a single special id.

### P0.6 Gate 0.2 architecture

- Did: loaded `Olmo-3-7B-Think-DPO@7b18bf927b430ff06376fdfa5610eb3b1b6a5c38` in bf16 on MPS (`scripts/00_arch.py`).
- Passed: 32 layers, d_model 4096, 32 KV heads, intermediate 11008, vocab 100278, untied embeddings, sliding window 4096, `layer_types` = (3× sliding_attention, 1× full_attention)×8, attention `sdpa`. `final_norm` is `Olmo3RMSNorm` with no bias. `lm_head` is `nn.Linear` with no bias. Block norms are `post_attention_layernorm`, `post_feedforward_layernorm`, plus `q_norm`/`k_norm`.
- Failed: nothing.
- Decided: final-norm has no bias and the LM head is a plain matrix, so the restricted lens cotangent `g ⊙ w_v` is valid. q_norm/k_norm stay out of the LRP patch, matching jpp.

### P0.6 Gate 0.3 throughput

- Did: `scripts/00_bench.py`. Best of 5 bf16 matmuls after warm-up, plus one 7B forward+backward at batch 8×128 with input grads only.
- Passed: torch 30.05 / 29.31 TFLOPS, MLX 30.54 / 31.01 TFLOPS (ratios 1.02 and 1.06). Forward+backward 2.00 s.
- Failed: nothing.
- Decided: MLX is not ≥ 2× on matmul, and the plan already keeps the fitter on torch-MPS. MLX stays the CoT-generation backend only. No MLX fitter port.

### P1 task generator

- Did: generated dev/pilot/main × nested/word, 50/200/300 per k. `TEMPLATE_NUMBERS` from the DPO chat template is `{2, 4}` (`2024` splits into `2` + `202` + `4`). `mul_symbol` is `×`.
- Passed: 6600 rows, 6600 unique format-independent keys, 0 constraint violations, 0 misaligned variant prompts, k=1 has no variants. Counts match. Unit tests pass.
- Failed: balance gate is non-blocking and fails for most k≥5 cells, and for k=1 main (`*` share 0.03 nested / 0.00 word). k=1 has only ~9800 legal chains; dev and pilot take the scarce `*` chains first, so main is the remainder. Higher k under-represents `*` because a `*` step also crowds a decile, and the max-deviation greedy will not take it.
- Decided: keep the specified generation order and report the gate. Null variants after 200 resamples: 1384/16500 patch sources, 396 controls, 187 swaps. Items are kept. Rebuild of `10_make_tasks.py` reproduced `data/MANIFEST.json` byte-for-byte (`match True`).

### P3.1 LRP mode

- Did: `src/s2/lrp.py`. Olmo3 copies jpp rlens (LN outside `self_attn`, identity+half on `Olmo3MLP`). Llama uses the same walker on Llama classes, with the stock weight-after-cast norm.
- Passed: `tests/test_lrp.py`. Logits with LRP on vs off are bitwise equal on tiny Olmo3 and Llama. Olmo3 input-gradients match `apply_lrp_rules_for_mode(..., "rlens")` with relative error < 1e-6. Patch names match, and `q_norm`/`k_norm` are not patched.
- Failed: a direct `z * sigmoid(z)` rewrite differed from `F.silu` by ~1e-7. The identity rule now returns `F.silu` in the forward and the detached-sigmoid gradient in the backward, so both gates hold.
- Decided: that forward correction is required by the bit-identity gate and does not drop the jpp gradient match.

### P0.6 Gate 0.4 attention — stop

- Did: `00_attn.py` then `00_attn_dtype.py`. Reference is CPU fp32 sdpa. Candidates are MPS bf16 sdpa, chunked sdpa, and chunked eager.
- Passed: 512-token spot check (min cosine 0.99892, argmax 0.9941). At 4096 and 8192 the median cosine is ≥ 0.99992 and fewer than 0.2% of positions are under 0.995.
- Failed: no path meets every-position cosine ≥ 0.995 and argmax ≥ 99% against CPU fp32 at 4096, so none were scored as passing at 8192. CPU bf16 versus CPU fp32 fails the same bar (8192 min cosine 0.956, argmax 0.987). MPS bf16 versus CPU bf16 at 8192 is min cosine 0.934, argmax 0.989, median 0.99992.
- Decided: stop and ask. This is not the silent MPS score-matrix corruption (that would not leave a 0.9999 median). Recommendation: use path (a) sdpa for Phase 6 and record the measured floor. Short-sequence work (fit at 128, readouts) does not depend on this.

### P0.6 Gate 0.5 memory

- Did: `00_memory.py` on 7B-DPO bf16 MPS. Peaks from `torch.mps.driver_allocated_memory()`.
- Passed: load 15.61 GB, forward+backward 8×128 24.24 GB, forward 1×16384 29.64 GB. All fit in 48 GB.
- Failed: nothing.
- Decided: the restricted fit at batch 8×128 has room. A 16k forward fits alone; do not overlap it with another model load.
