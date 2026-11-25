import numpy as np

from s2.codi_items import gate4_items


def test_gate4_items_match_the_two_step_formula():
    rows = gate4_items(20)
    assert len(rows) == 40
    assert sum(row["sign"] == "+" for row in rows) == 20
    for row in rows:
        s1 = row["x"] + row["y"] if row["sign"] == "+" else row["x"] - row["y"]
        assert row["s1"] == s1
        assert row["s2"] == s1 * row["z"]
        assert row["answer"] == row["s2"] + row["s1"]
        assert len({row["x"], row["y"], row["z"], row["s1"], row["s2"], row["answer"]}) == 6
        assert 1 <= row["answer"] <= 99


def test_source_item_keeps_z_and_sets_s1():
    from s2.codi_items import source_item

    target = {"sign": "+", "z": 3, "s1": 8}
    source = source_item(target, b=5, rng=np.random.default_rng(0))
    assert source is not None
    assert source["s1"] == 5 and source["z"] == 3
    assert source["answer"] == 5 * 3 + 5
