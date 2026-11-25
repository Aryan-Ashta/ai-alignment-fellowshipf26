"""Lens-swap clamp. Matches jpp ``Clamp._edit_selected``: ``h + V(c* - c)``."""

from __future__ import annotations

import torch


def unit_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Columns are unit vectors. ``[d, 2]``."""
    pair = torch.stack([a.float(), b.float()], dim=1)
    return pair / pair.norm(dim=0, keepdim=True).clamp_min(1e-12)


def clamp_swap(hidden: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """``hidden`` is ``[P, d]``. Returns the edited residual in the same dtype."""
    atoms = unit_pair(a, b).to(hidden.device)
    pinv = torch.linalg.pinv(atoms)
    coordinates = hidden.float() @ pinv.T
    targets = coordinates.clone()
    targets[:, 0] = scale * coordinates[:, 1]
    targets[:, 1] = scale * coordinates[:, 0]
    edited = hidden.float() + (targets - coordinates) @ atoms.T
    return edited.to(dtype=hidden.dtype)
