# Handoff

Stop point: Gate 3 failed. Aryan said stop, write `findings.md`, and write this file. Do not fit 64 prompts. Do not start readout on `olmo3_7b.dpo` until Aryan reverses that.

HEAD: `9008362` on `master`, pushed to `https://github.com/Aryan-Ashta/ai-alignment-fellowshipf26.git`. Project root is `stage2-whitebox/` inside that repo, not `~/research/stage2-whitebox/`.

## Read first

1. `findings.md` for the cosine numbers.
2. `logs/AGENT_LOG.md` for the session log.
3. `stage2_whitebox_agent_plan.md` at the repo root. It is still untracked. Do not commit it unless Aryan asks. It is his plan file.

## On disk, not in git

`**/*.pt` is gitignored.

- `lenses/olmo3_7b.dpo/restricted_rows.pt` — 32-prompt lens, 531 MB, 2026-11-25 20:36.
- `lenses/olmo3_7b.dpo/fit_accum.pt` — resume state, `next_prompt` 32, commit stamp from the run.
- `lenses/olmo3_7b.dpo/p16/restricted_rows.pt` — 16-prompt lens, 531 MB, 2026-11-25 21:45.
- `lenses/olmo3/candidates.json` — 985 readout ids (100 number tokens, then dev items, then 500 distractors) and 110 swap rows.
- `scratch/codi/codi_merged/` — merged CODI weights, if the download finished. Not a Gate 4 result.

## Commands that already ran

```bash
uv run python scripts/33_check_fit.py --model olmo3_7b --ckpt dpo
uv run python scripts/33_convergence.py \
  --small lenses/olmo3_7b.dpo/p16/restricted_rows.pt \
  --large lenses/olmo3_7b.dpo/restricted_rows.pt
```

The check printed `min cosine 0.7851 rows 30535` and exited 3.

## If Aryan later says to continue

- Another fit must use `--tag`. Reusing `fit_accum.pt` with `--prompts 16` would divide the 32-prompt sum by 16.
- Short WikiText records are skipped in `scripts/31_fit.py`. Do not revert that. A 600-character record can be under 128 tokens.
- Long jobs call `require_ac_power()`. On this Mac that raises unless `pmset -g batt` says `AC Power`. On Linux the check is skipped because there is no `pmset`.
- Do not commit while a fit is between accumulator saves. The accumulator stores `code_commit()`, and a newer HEAD makes the resume check refuse.
- Launch with `caffeinate -dimsu`. One 7B fit at a time. Do not share the GPU with Gate 4 or readout.

## Scripts that exist and have not been run on 7B

| step | script | needs |
| --- | --- | --- |
| router | `scripts/30_router.py` | AC, 7B |
| expert weights | `scripts/32_experts.py` | a logit tensor that does not exist yet |
| readout | `scripts/40_readouts.py` | the lens file, AC |
| no-CoT | `scripts/50_nocot.py` | AC |
| swap | `scripts/51_swap.py` | lens + `nocot.parquet` |
| patch | `scripts/52_patch.py` | writes `patch.parquet`, not `patching.parquet` |
| CODI Gate 4 | `scripts/54_codi_gate4.py` | AC, merged CODI model |
| CoT | `scripts/60_cot.py` | AC |
| pilot status | `scripts/70_pilot.py` | exits 2 until fit, readout, nocot, swap, and patch exist |
| analysis | `scripts/95_analysis.py` | refuses a depth table until `prereg/thresholds.yaml` exists |
| 32B cost | `scripts/90_cost_estimate.py` | refuses dollars without `--usd-per-gpu-hour` |

`cloud/run_*.sh` cd to `stage2-whitebox` and pass `--steps hooks,router,fit,readout,nocot,swap,patch,cot`. They have not been launched. There is no GPU rate, so no dollar figure.

## Gates already passed, on this machine

- Gate 0 preflight: tokenizer, Olmo-3-7B architecture, MPS matmul, attention dtype. Path (a) SDPA is frozen in `configs/gen.yaml`. MPS bf16 misses the fp32 bar at 4096 and 8192.
- Gate 1: task parquet, constraints, determinism. `TEMPLATE_NUMBERS` is `(2, 4)`.
- Gate 2: hook identity on 7B-DPO and on the merged CODI model. Max abs logit diff 0.
- Restricted VJP matches a per-position sum to relative error < 1e-5 on a tiny model. That does not override the failed 7B convergence check.
- Astra arm skips. `configs/gen.yaml` `astra_instruction` is null. Do not invent Sunny's wording.

## Open choices that are Aryan's

1. Gate 3: stop stands. 64-prompt fit and readout are both blocked.
2. Cloud GPU rate for the 32B estimate.
3. CoT-Control wording, if the Astra arm is ever turned on.
4. Whether random distractors stay in the candidate set. They are already in the 985 readout rows.
5. `stage2_whitebox_agent_plan.md` is untracked on purpose.
