import pandas as pd

from s2.analysis.bootstrap import accuracy_table


def test_unleaked_accuracy_drops_leaked_items(tmp_path):
    nocot = pd.DataFrame(
        [
            {"format": "nested", "k": 1, "correct": True, "leaked": False},
            {"format": "nested", "k": 1, "correct": False, "leaked": True},
        ]
    )
    table = accuracy_table(nocot, tmp_path / "accuracy.csv")
    row = table.iloc[0]
    assert row["accuracy_all"] == 0.5
    assert row["accuracy_unleaked"] == 1.0
    assert row["leakage"] == 0.5
