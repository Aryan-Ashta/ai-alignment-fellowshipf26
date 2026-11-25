import pandas as pd

from s2.analysis.depth import Thresholds, depth_ci


def _thresholds() -> Thresholds:
    return Thresholds(
        k_max=4,
        formats=("nested",),
        t_acc=0.5,
        lens="jpp",
        rank_max=5,
        positions=(-1,),
        layers="all",
        min_item_rate=0.8,
        t_share=0.5,
        band_rule="fixed",
        scale=1.0,
    )


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nocot, readouts, swaps = [], [], []
    for k, n_correct in ((1, 20), (2, 20), (3, 20), (4, 0)):
        for i in range(20):
            item = f"k{k}-{i}"
            correct = i < n_correct
            nocot.append({"item_id": item, "k": k, "format": "nested", "correct": correct, "leaked": False})
            for kind, rank in (("intermediate", 1), ("null", 50)):
                if k == 1 and kind == "intermediate":
                    continue
                readouts.append(
                    {
                        "item_id": item,
                        "k": k,
                        "format": "nested",
                        "target_kind": kind,
                        "step": 1,
                        "layer": 8,
                        "pos_offset": -1,
                        "lens": "jpp",
                        "rank_num": rank if correct else 80,
                    }
                )
            swaps.append(
                {
                    "item_id": item,
                    "k": k,
                    "format": "nested",
                    "variant": "main",
                    "scale": 1.0,
                    "success": correct,
                }
            )
    return pd.DataFrame(nocot), pd.DataFrame(readouts), pd.DataFrame(swaps)


def test_synthetic_depth_is_3_and_ci_contains_3():
    nocot, readouts, swaps = _fixture()
    point, low, high = depth_ci(nocot, readouts, swaps, _thresholds(), n=200, seed=0)
    assert point == 3
    assert low <= 3 <= high
