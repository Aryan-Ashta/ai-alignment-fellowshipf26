from s2.analysis.narrative import write_narrative


def test_narrative_does_not_invent_a_comparison(tmp_path):
    text = write_narrative({}, tmp_path / "STAGE2_RESULTS.md")
    assert "No measured comparison is available yet." in text
    assert "number tokens only" in text
