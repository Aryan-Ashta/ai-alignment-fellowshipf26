import pandas as pd

from s2.analysis.cot_table import cot_accuracy


def test_cot_accuracy_is_parsed_equals_answer(tmp_path):
    records = pd.DataFrame(
        [
            {"k": 1, "answer": 4, "parsed": 4},
            {"k": 1, "answer": 9, "parsed": None},
        ]
    )
    table = cot_accuracy(records, tmp_path / "cot.csv")
    assert table.iloc[0]["accuracy"] == 0.5
