"""Resolve Stage 2 checkpoints to commit SHAs and weight-file hashes.

Writes ``configs/models.yaml`` and ``logs/preflight/checkpoint_identity.md``.
Loading code must use ``revision=<sha>`` from that file, never a branch name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import write_manifest  # noqa: E402

# Live API must return these. They are the 2026-11-25 identity facts.
EXPECTED_SHA = {
    ("allenai/Olmo-3-7B-Think-DPO", "main"): "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38",
    ("allenai/Olmo-3-7B-Think", "main"): "d97e442d7cc678210054dbcc9b440894d62c89a4",
    ("allenai/Olmo-3-7B-Think", "step_0150"): "ce0ae8ce5ea7a89856a75d662de888deebc39105",
    ("allenai/Olmo-3-7B-Think", "step_0700"): "d8d03802a525af6e346859adb9f39fad216f1bb8",
    ("allenai/Olmo-3-7B-Think", "step_1375"): "031240693eb33d302cfa8d2df76af15d2da4b579",
    ("allenai/Olmo-3-32B-Think-DPO", "main"): "97604023f2602a7dfe6d31b1fc229985e0002022",
    ("allenai/Olmo-3-32B-Think", "step_750"): "f395de048595db659a2562913961b2ec5585eb42",
    ("allenai/Olmo-3.1-32B-Think", "main"): "97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d",
    ("allenai/Olmo-3.1-32B-Think", "step_0250"): "5c349dde8eae4760432cb856bfb53fbc77884705",
    ("allenai/Olmo-3.1-32B-Think", "step_0750"): "418fb27467ffd7f54b4eca285ea59b2b426fee07",
    ("allenai/Olmo-3.1-32B-Think", "step_2300"): "29884259e8c87406883e817e0628fa5ce9b529c3",
    ("bcywinski/codi_llama1b-answer_only", "main"): "315ec5cb5f9a071f950edb8455ca5c9a5f59691e",
}

CODI_BASE_REPO = "unsloth/Llama-3.2-1B-Instruct"
CODI_BASE_SAFETENSORS_BYTES = 2471645608

CHECKPOINTS = {
    "olmo3_7b": {
        "dpo": ("allenai/Olmo-3-7B-Think-DPO", "main", 0, "dpo"),
        "early": ("allenai/Olmo-3-7B-Think", "step_0150", 150, "step_0150"),
        "mid": ("allenai/Olmo-3-7B-Think", "step_0700", 700, "step_0700"),
        "final": ("allenai/Olmo-3-7B-Think", "main", 1375, "main"),
    },
    "olmo3_32b": {
        "dpo": ("allenai/Olmo-3-32B-Think-DPO", "main", 0, "dpo"),
        "early": ("allenai/Olmo-3.1-32B-Think", "step_0250", 250, "step_0250"),
        "mid": ("allenai/Olmo-3-32B-Think", "step_750", 750, "step_750"),
        "final": ("allenai/Olmo-3.1-32B-Think", "main", 2300, "main"),
    },
}


def branch_targets(api: HfApi, repo: str) -> dict[str, str]:
    refs = api.list_repo_refs(repo)
    return {ref.name: ref.target_commit for ref in refs.branches}


def resolve_branch(api: HfApi, repo: str, branch: str) -> str:
    targets = branch_targets(api, repo)
    if branch not in targets:
        raise SystemExit(f"{repo} has no branch {branch!r}; have {sorted(targets)}")
    sha = targets[branch]
    expected = EXPECTED_SHA.get((repo, branch))
    if expected is not None and sha != expected:
        raise SystemExit(f"{repo}@{branch} is {sha}, expected {expected}")
    return sha


def weight_rows(api: HfApi, repo: str, revision: str) -> list[dict[str, str | int | None]]:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise SystemExit(f"{repo}@{revision} model_info.sha={info.sha}")
    rows: list[dict[str, str | int | None]] = []
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if "tokenizer" in name:
            continue
        if not (name.endswith(".safetensors") or name.endswith(".bin")):
            continue
        lfs = sibling.lfs
        if lfs is None or not lfs.sha256:
            raise SystemExit(f"{repo}@{revision} weight {name} has no LFS sha256")
        rows.append({"file": name, "sha256": lfs.sha256, "size": lfs.size})
    if not rows:
        raise SystemExit(f"{repo}@{revision} has no weight files")
    return sorted(rows, key=lambda row: str(row["file"]))


def sha_list(rows: list[dict]) -> list[str]:
    return [str(row["sha256"]) for row in rows]


def entry(repo: str, branch: str, revision: str, rl_step: int | None, label: str, rows: list[dict]) -> dict:
    return {
        "repo": repo,
        "branch": branch,
        "revision": revision,
        "rl_step": rl_step,
        "label": label,
        "shard_sha256": rows,
    }


def identical(left: list[str], right: list[str]) -> bool:
    return left == right and len(left) > 0


def main() -> None:
    api = HfApi()
    models: dict = {}
    resolved: dict[tuple[str, str], str] = {}
    weights: dict[tuple[str, str], list[dict]] = {}

    needed = set(EXPECTED_SHA)
    needed.add((CODI_BASE_REPO, "main"))
    for family in CHECKPOINTS.values():
        for repo, branch, _step, _label in family.values():
            needed.add((repo, branch))

    for repo, branch in sorted(needed):
        sha = resolve_branch(api, repo, branch)
        resolved[(repo, branch)] = sha
        print(f"{repo}@{branch} {sha}", flush=True)
        weights[(repo, branch)] = weight_rows(api, repo, sha)

    for family, slots in CHECKPOINTS.items():
        models[family] = {}
        for slot, (repo, branch, rl_step, label) in slots.items():
            sha = resolved[(repo, branch)]
            models[family][slot] = entry(repo, branch, sha, rl_step, label, weights[(repo, branch)])

    codi_repo, codi_branch = "bcywinski/codi_llama1b-answer_only", "main"
    models["codi"] = entry(
        codi_repo,
        codi_branch,
        resolved[(codi_repo, codi_branch)],
        None,
        "codi",
        weights[(codi_repo, codi_branch)],
    )
    models["codi_base"] = entry(
        CODI_BASE_REPO,
        "main",
        resolved[(CODI_BASE_REPO, "main")],
        None,
        "codi_base",
        weights[(CODI_BASE_REPO, "main")],
    )

    base_rows = weights[(CODI_BASE_REPO, "main")]
    st = [row for row in base_rows if row["file"] == "model.safetensors"]
    if len(st) != 1 or st[0]["size"] != CODI_BASE_SAFETENSORS_BYTES:
        raise SystemExit(
            f"{CODI_BASE_REPO} model.safetensors size {st!r}, expected {CODI_BASE_SAFETENSORS_BYTES}"
        )

    seven_main = sha_list(weights[("allenai/Olmo-3-7B-Think", "main")])
    seven_last = sha_list(weights[("allenai/Olmo-3-7B-Think", "step_1375")])
    mid_31 = sha_list(weights[("allenai/Olmo-3.1-32B-Think", "step_0750")])
    mid_30 = sha_list(weights[("allenai/Olmo-3-32B-Think", "step_750")])
    final_31 = sha_list(weights[("allenai/Olmo-3.1-32B-Think", "main")])
    final_last = sha_list(weights[("allenai/Olmo-3.1-32B-Think", "step_2300")])

    checks = [
        (
            "7B Think main vs step_1375",
            identical(seven_main, seven_last),
            False,
            seven_main,
            seven_last,
        ),
        (
            "32B Olmo-3.1 step_0750 vs Olmo-3 step_750",
            identical(mid_31, mid_30),
            True,
            mid_31,
            mid_30,
        ),
        (
            "32B Olmo-3.1 main vs step_2300",
            identical(final_31, final_last),
            True,
            final_31,
            final_last,
        ),
    ]
    failures = [name for name, got, want, _a, _b in checks if got != want]
    lines = [
        "# Checkpoint identity",
        "",
        "Resolved by `scripts/00_resolve_checkpoints.py` via `HfApi.list_repo_refs` and `model_info(files_metadata=True)`.",
        "Weight identity compares sorted LFS sha256 lists of `*.safetensors` / `*.bin` files.",
        "",
        "## Revisions",
        "",
        "| repo | branch | revision |",
        "|---|---|---|",
    ]
    for (repo, branch), sha in sorted(resolved.items()):
        lines.append(f"| `{repo}` | `{branch}` | `{sha}` |")
    lines.extend(["", "## Weight identity", ""])
    for name, got, want, left, right in checks:
        status = "PASS" if got == want else "FAIL"
        lines.append(f"- **{status}** {name}: identical={got}, expected identical={want}.")
        lines.append(f"  - left: `{left}`")
        lines.append(f"  - right: `{right}`")
    lines.extend(
        [
            "",
            "## Plot steps",
            "",
            "- 7B `final` is `Olmo-3-7B-Think@main`. Bytes differ from `step_1375`. `rl_step` is 1375 and `label` is `main`, so plots sit at the end of the published RL grid without claiming byte identity.",
            "- 32B `final` is `Olmo-3.1-32B-Think@main`, byte-identical to `step_2300`. `rl_step` is 2300 and `label` is `main`.",
            "- 32B `mid` is `Olmo-3-32B-Think@step_750`, byte-identical to `Olmo-3.1-32B-Think@step_0750`.",
            f"- `codi_base` is `{CODI_BASE_REPO}@main` (`{resolved[(CODI_BASE_REPO, 'main')]}`). `model.safetensors` is {CODI_BASE_SAFETENSORS_BYTES} bytes.",
            "",
        ]
    )
    out_dir = ROOT / "logs" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = out_dir / "checkpoint_identity.md"
    identity.write_text("\n".join(lines))
    config_path = ROOT / "configs" / "models.yaml"
    config_path.write_text(yaml.safe_dump(models, sort_keys=False))
    write_manifest(out_dir, script="00_resolve_checkpoints.py")
    print(f"wrote {config_path}")
    print(f"wrote {identity}")
    if failures:
        raise SystemExit(f"identity checks failed: {failures}")


if __name__ == "__main__":
    main()
