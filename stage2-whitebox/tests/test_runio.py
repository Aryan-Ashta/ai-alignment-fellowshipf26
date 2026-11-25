import json
from pathlib import Path

import pytest
import torch

from s2.runio import (
    REFUSAL,
    atomic_torch_save,
    manifest_fields,
    reports_ac_power,
    require_ac_power,
    sha256_file,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ac_power_parser():
    assert reports_ac_power("Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged\n")
    assert not reports_ac_power(
        "Now drawing from 'Battery Power'\n -InternalBattery-0 71%; discharging\n"
    )
    require_ac_power("Now drawing from 'AC Power'\n")
    with pytest.raises(RuntimeError, match=REFUSAL):
        require_ac_power("Now drawing from 'Battery Power'\n")


def test_manifest_has_spec_fields(tmp_path: Path):
    path = write_manifest(tmp_path, model_id="example", revision="a" * 40, seeds={"k": 1})
    payload = json.loads(path.read_text())
    for key in (
        "code_commit",
        "pyproject_lock_hash",
        "versions",
        "os",
        "device_name",
        "model_id",
        "revision",
        "item_file_sha256",
        "fitting_corpus_sha256",
        "seeds",
        "wall_time_s",
        "peak_memory",
        "thermal_note",
    ):
        assert key in payload
    assert payload["lock_file"] == "uv.lock"
    assert payload["pyproject_lock_hash"] == sha256_file(ROOT / "uv.lock")
    assert payload["model_id"] == "example"
    assert payload["versions"]["torch"] == torch.__version__
    assert len(payload["code_commit"]) == 40


def test_manifest_fields_match_writer(tmp_path: Path):
    extra = {"model_id": "m", "wall_time_s": 1.5}
    assert manifest_fields(**extra)["wall_time_s"] == 1.5


def test_atomic_torch_save_roundtrip(tmp_path: Path):
    path = tmp_path / "nested" / "state.pt"
    atomic_torch_save({"x": torch.tensor([1, 2, 3])}, path)
    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["x"], torch.tensor([1, 2, 3]))
    assert not list(path.parent.glob("*.tmp.*"))
