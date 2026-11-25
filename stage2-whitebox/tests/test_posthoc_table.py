import pandas as pd

from s2.analysis.posthoc_table import lead_table


def test_lead_table_uses_ranks_before_the_write(tmp_path):
    ranks = pd.DataFrame(
        [
            {"item_id": "a", "step": 1, "position": 0, "rank": 9, "first_write": 2},
            {"item_id": "a", "step": 1, "position": 1, "rank": 3, "first_write": 2},
        ]
    )
    table = lead_table(ranks, tmp_path / "lead.csv")
    assert int(table.iloc[0]["lead"]) == 1
