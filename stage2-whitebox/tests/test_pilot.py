from s2.pilot import missing_steps


def test_missing_steps_names_only_absent_files(tmp_path):
    key = "olmo3_7b.dpo"
    path = tmp_path / "results" / key
    path.mkdir(parents=True)
    (path / "nocot.parquet").write_bytes(b"")
    assert missing_steps(tmp_path, key) == ["fit", "readouts", "swaps", "patch"]
