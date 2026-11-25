"""Per-checkpoint driver. Long steps refuse unless the Mac is on AC.

A step reruns when ``STEP_DONE/<step>.json`` is missing or records a different
code commit. ``hooks`` runs in process. Router, fit, readout, no-CoT, swap,
patch and CoT call the matching script. Expert weights stay unwired until a
logit tensor exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.models import load  # noqa: E402
from s2.runio import code_commit, require_ac_power  # noqa: E402

LONG_STEPS = {"router", "fit", "weights", "readout", "nocot", "swap", "patch", "cot"}


def marker(results: Path, step: str) -> Path:
    return results / "STEP_DONE" / f"{step}.json"


def step_current(results: Path, step: str) -> bool:
    path = marker(results, step)
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    return payload.get("code_commit") == code_commit()


def mark(results: Path, step: str) -> None:
    path = marker(results, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"code_commit": code_commit(), "step": step}, indent=2) + "\n")


def run_hooks(model_key: str, device: str, attn: str) -> None:
    model = load(model_key, device=device, attn=attn)
    ids = torch.randint(0, 100, (1, 8), device=device)
    bare = model.hf(input_ids=ids).logits
    with model.capture([0, model.n_layers - 1]) as stored:
        captured = model.hf(input_ids=ids).logits
    if not torch.equal(bare, captured):
        raise SystemExit("capture hooks changed logits")
    with model.replace({0: lambda h: h}):
        replaced = model.hf(input_ids=ids).logits
    if not torch.equal(bare, replaced):
        raise SystemExit("identity replace changed logits")
    readout = model.final_readout(stored[model.n_layers - 1])
    if not torch.equal(readout, bare):
        raise SystemExit("final_readout != logits")
    print(f"hooks ok {model_key} {model.revision}")


def step_argv(step: str, model: str, ckpt: str, device: str, split: str) -> list[str]:
    scripts = {
        "router": "30_router.py",
        "fit": "31_fit.py",
        "readout": "40_readouts.py",
        "nocot": "50_nocot.py",
        "swap": "51_swap.py",
        "patch": "52_patch.py",
        "cot": "60_cot.py",
    }
    if step not in scripts:
        raise SystemExit(f"{step} is not wired yet")
    argv = [sys.executable, str(ROOT / "scripts" / scripts[step]), "--model", model, "--ckpt", ckpt, "--device", device]
    if step in {"nocot", "swap", "cot"}:
        argv.extend(["--split", split])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--steps", default="hooks")
    args = parser.parse_args()
    if args.ckpt is None and args.model != "codi":
        raise SystemExit("--ckpt is required")
    key = f"{args.model}.{args.ckpt}" if args.ckpt else args.model
    results = Path(args.results_root) / key
    for step in args.steps.split(","):
        if step_current(results, step):
            print(f"skip {step}")
            continue
        if step in LONG_STEPS or step == "nocot":
            require_ac_power()
            import subprocess

            subprocess.check_call(step_argv(step, args.model, args.ckpt or "dpo", args.device, args.split))
        elif step == "hooks":
            run_hooks(key, args.device, args.attn)
        else:
            raise SystemExit(f"unknown step {step}")
        mark(results, step)


if __name__ == "__main__":
    main()
