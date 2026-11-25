import pandas as pd

from s2.analysis.curves import mean_curve


def test_curve_is_the_mean_success(tmp_path):
    frame = pd.DataFrame(
        [
            {"layer": 0, "scale": 1.0, "variant": "main", "success": True},
            {"layer": 0, "scale": 1.0, "variant": "main", "success": False},
        ]
    )
    curve = mean_curve(frame, tmp_path / "swaps.csv")
    assert curve.iloc[0]["success"] == 0.5
