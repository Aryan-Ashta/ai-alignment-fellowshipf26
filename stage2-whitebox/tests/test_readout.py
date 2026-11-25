import torch

from s2.readout import rank_among


def test_rank_is_one_based_and_ties_favour_the_target():
    scores = torch.tensor([1.0, 3.0, 3.0, 2.0])
    assert rank_among(scores, 1) == 1
    assert rank_among(scores, 2) == 1
    assert rank_among(scores, 3) == 3
    assert rank_among(scores, 0) == 4
