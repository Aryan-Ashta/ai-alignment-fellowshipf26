import matplotlib
import pandas as pd

matplotlib.use("Agg")

from s2.analysis.plots import plot_swap


def test_swap_plot_writes_a_png(tmp_path):
    csv = tmp_path / "swaps.csv"
    pd.DataFrame({"layer": [0, 4], "success": [0.1, 0.4], "variant": ["main", "main"]}).to_csv(csv, index=False)
    out = tmp_path / "swaps.png"
    plot_swap(csv, out)
    assert out.stat().st_size > 0
