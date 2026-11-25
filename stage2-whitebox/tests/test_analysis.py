import pandas as pd

from s2.analysis.depth import Thresholds
from s2.analysis.plots import plot_depth
from s2.analysis.tables import depth_table


def test_depth_table_and_plot(tmp_path):
    nocot = pd.DataFrame(
        [{"item_id": "a", "k": 1, "format": "nested", "correct": True, "leaked": False}]
    )
    readouts = pd.DataFrame(
        columns=["item_id", "k", "format", "target_kind", "step", "layer", "pos_offset", "lens", "rank_num"]
    )
    swaps = pd.DataFrame(columns=["item_id", "k", "format", "variant", "scale", "success"])
    thresholds = Thresholds(1, ("nested",), 0.5, "jpp", 5, (-1,), "all", 0.8, 0.5, "fixed", 1.0)
    table = depth_table(
        [{"model": "olmo3_7b", "revision": "a" * 40, "rl_step": 0, "nocot": nocot, "readouts": readouts, "swaps": swaps}],
        thresholds,
        tmp_path / "depth_by_checkpoint.csv",
    )
    assert int(table.iloc[0]["depth"]) == 1
    plot_depth(tmp_path / "depth_by_checkpoint.csv", tmp_path / "depth.png")
    assert (tmp_path / "depth.png").is_file()
