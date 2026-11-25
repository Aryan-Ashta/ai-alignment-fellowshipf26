import torch

from s2.kmeans_router import fit_and_save, load_router


def test_router_assignments_survive_save(tmp_path):
    torch.manual_seed(0)
    acts = {0: torch.randn(80, 64), 1: torch.randn(80, 64)}
    path = tmp_path / "router.pt"
    fitted = fit_and_save(acts, path)
    loaded = load_router(path)
    for layer, residual in acts.items():
        assert torch.equal(fitted.assign(residual, layer), loaded.assign(residual, layer))
