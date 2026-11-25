"""Per-checkpoint router. The clustering itself is jpp's torch implementation."""

from __future__ import annotations

from pathlib import Path

import torch
from workspace_lens.routing.router import ActivationRouterCollection


def new_router() -> ActivationRouterCollection:
    return ActivationRouterCollection(
        num_clusters=8,
        projection_dim=64,
        seed=0,
        kmeans_max_iterations=100,
    )


def fit_and_save(activations: dict[int, torch.Tensor], path: str | Path) -> ActivationRouterCollection:
    router = new_router()
    router.fit(activations)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    router.save(str(path))
    return router


def load_router(path: str | Path) -> ActivationRouterCollection:
    return ActivationRouterCollection.load(str(path))
