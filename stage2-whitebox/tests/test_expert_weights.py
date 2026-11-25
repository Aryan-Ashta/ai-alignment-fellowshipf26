import torch

from s2.expert_weights import learn_weights


def test_learned_weights_are_unit_l2():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 6)
    candidate_ids = [{"a": [0, 1]}, {"b": [2]}]
    weights = learn_weights(logits, candidate_ids, ["multihop", "typo"])
    assert weights.shape == (4,)
    assert torch.allclose(weights.norm(), torch.tensor(1.0), atol=1e-5)
