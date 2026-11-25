import torch

from s2.swaps import clamp_swap, unit_pair


def test_clamp_exchanges_coordinates_at_the_requested_scale():
    torch.manual_seed(0)
    a = torch.randn(16)
    b = torch.randn(16)
    hidden = torch.randn(4, 16)
    edited = clamp_swap(hidden, a, b, scale=2.0)
    atoms = unit_pair(a, b)
    before = hidden @ torch.linalg.pinv(atoms).T
    after = edited @ torch.linalg.pinv(atoms).T
    assert torch.allclose(after[:, 0], 2 * before[:, 1], atol=1e-5)
    assert torch.allclose(after[:, 1], 2 * before[:, 0], atol=1e-5)
