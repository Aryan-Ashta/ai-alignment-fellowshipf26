# Stage 2 white-box pipeline

Measures internal step depth on Olmo-3 Think checkpoints: how many arithmetic steps the model completes without a chain of thought while intermediate results are readable (candidate-restricted J++ lens, R-lens, logit lens) and causally used (lens swap, activation patching).

Spec: `../stage2_whitebox_agent_plan.md` (not edited by this tree). Frozen choices in that file's §1.3 are not changed here.

Working directory is this folder. Commands run as `uv run python scripts/<name>.py`.

Long jobs call `require_ac_power()` and are launched as `caffeinate -dimsu uv run python scripts/<x>.py`.
