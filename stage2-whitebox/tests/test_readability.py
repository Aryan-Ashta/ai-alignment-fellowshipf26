import pandas as pd

from s2.analysis.readability import readability_curve


def test_readability_curve_is_the_hit_rate(tmp_path):
    frame = pd.DataFrame(
        [
            {"lens": "logit", "k": 1, "layer": 0, "hit": True},
            {"lens": "logit", "k": 1, "layer": 0, "hit": False},
        ]
    )
    curve = readability_curve(frame, tmp_path / "readability.csv")
    assert curve.iloc[0]["hit"] == 0.5
