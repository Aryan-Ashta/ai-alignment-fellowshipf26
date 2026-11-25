from s2.posthoc import lead_time


def test_lead_time_is_positions_before_the_first_write():
    assert lead_time([9, 4, 2, 1], first_write=3, rank_max=5) == 2
    assert lead_time([9, 9, 9], first_write=2, rank_max=5) is None
    assert lead_time([1], first_write=None, rank_max=5) is None
