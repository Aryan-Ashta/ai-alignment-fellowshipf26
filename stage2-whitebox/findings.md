# Findings

Date: 2026-11-25. Machine: MacBook Pro, Apple M5 Pro, 48 GB, macOS. Repo HEAD at the stop: `9008362`.

## Stop

Gate 3 failed on `olmo3_7b.dpo`. Aryan chose stop. Do not start a 64-prompt fit. Do not start readout, no-CoT, swap, or patching on this lens until that choice is reversed.

## Convergence

Compared `lenses/olmo3_7b.dpo/p16/restricted_rows.pt` (16 prompts) with `lenses/olmo3_7b.dpo/restricted_rows.pt` (32 prompts). Both are readout `(985, 31, 4096)` and swap `(110, 31, 4096)`. The check is cosine of every readout row, threshold 0.99.

| quantile | cosine |
| --- | --- |
| min | 0.7851 |
| 1% | 0.8477 |
| 5% | 0.8973 |
| median | 0.9788 |
| 95% | 0.9987 |
| max | 0.9996 |

21553 of 30535 rows are below 0.99. 7129 are below 0.95. 1643 are below 0.90.

Early layers move the most. Per-layer minimum cosine starts at 0.7975 (layer 0) and 0.7851 (layer 2). It crosses 0.99 only at layer 25 (0.9915) and stays there through layer 30 (0.9989).

`scripts/33_convergence.py` exited 3.

## What that means

32 WikiText prompts are not enough for the restricted readout rows to settle, except in the last few layers. The plan says a failed 16-vs-32 check means the prompt count is too small, and not to silently move to 64. A 64-prompt fit would be about 4 hours at the measured rate and is not started.

## Fit facts

- One prompt is about 4 minutes. 29 prompts took 7480 seconds before a short record aborted the run. The last 3 prompts of the resume took 782 seconds.
- A WikiText record of at least 600 characters can still encode to fewer than 128 tokens. Prompt index 29 was 123 tokens. The script now skips those records. The first 29 summed prompts are the first 29 records that did encode to 128.
- The 16-prompt fit is the first 16 of that same filtered stream, written under `--tag p16` so it cannot divide the 32-prompt accumulator by 16.
- Both `.pt` files are gitignored. They are on this machine only.

## What is not a finding

Readout, no-CoT accuracy, swaps, patching, and CODI Gate 4 have not been run. There is no evidence yet on those gates. The depth metric's synthetic fixture passed earlier. That is not a result on Olmo.
