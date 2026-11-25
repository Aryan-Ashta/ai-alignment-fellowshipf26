import pandas as pd

from s2.pilot_report import write_report


def test_report_quotes_measured_rates_and_marks_missing_files(tmp_path):
    key = "olmo3_7b.dpo"
    results = tmp_path / "results" / key
    results.mkdir(parents=True)
    pd.DataFrame({"correct": [True, False], "leaked": [False, True]}).to_parquet(results / "nocot.parquet")
    text = write_report(tmp_path, key, tmp_path / "PILOT_REPORT.md")
    assert "no-CoT accuracy: 0.500 n=2" in text
    assert "leakage: 0.500 n=2" in text
    assert "swap success: missing" in text
    assert "Do not start the main 7B study" in text
