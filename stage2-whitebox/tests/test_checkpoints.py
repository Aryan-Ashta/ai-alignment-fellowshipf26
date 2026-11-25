"""Checkpoint records are full SHAs. Loaders must not pass branch names."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
BRANCH_REVISION = re.compile(
    r"""from_pretrained\([^)]*revision\s*=\s*['"](?:main|step_\d+)['"]""",
    re.S,
)


def _entries(node):
    if isinstance(node, dict):
        if {"repo", "branch", "revision"} <= node.keys():
            yield node
        for value in node.values():
            yield from _entries(value)


def test_models_yaml_revisions_are_full_shas():
    data = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entries = list(_entries(data))
    assert len(entries) == 10
    for entry in entries:
        assert SHA.fullmatch(entry["revision"]), entry
        assert entry["revision"] != entry["branch"]
        assert entry["shard_sha256"]
        for row in entry["shard_sha256"]:
            assert SHA256.fullmatch(row["sha256"]), row["sha256"]


def test_no_branch_name_reaches_from_pretrained():
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    assert paths
    for path in paths:
        text = path.read_text()
        match = BRANCH_REVISION.search(text)
        assert match is None, f"{path}: {match.group(0)}"
