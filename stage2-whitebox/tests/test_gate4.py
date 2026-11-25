from s2.gate4 import patching_passes, readability_passes


def test_readability_requires_both_rates_and_order():
    assert readability_passes(0.60, 0.60, 1, 3)
    assert not readability_passes(0.59, 0.80, 1, 3)
    assert not readability_passes(0.80, 0.80, 4, 2)
    assert not readability_passes(0.80, 0.80, None, 2)


def test_patching_requires_source_success_and_a_quiet_control():
    assert patching_passes(0.50, 0.10)
    assert not patching_passes(0.49, 0.0)
    assert not patching_passes(0.80, 0.11)
