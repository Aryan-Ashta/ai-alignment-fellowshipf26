"""Restricted J++ / R-lens rows.

Estimator, matching ``third_party/jacobian-lens/jlens/fitting.py`` lines 97-210
and jpp ``expert_fitting.py`` per-position pooling:

For each prompt, place cotangent ``c_v`` at every kept target position of the
final block output (positions ``[16, seq_len-1)``) and 0 elsewhere. The
source gradient at position ``p`` is then ``sum_{p' in kept, p' >= p} dh_{L-1,p'} / dh_{l,p}``
applied to ``c_v``, because causality zeros earlier targets. The pooled row is
the unweighted mean of those source gradients over kept positions. The expert
row is the same mean over positions assigned to that expert. An expert with
fewer than 50 kept positions falls back to the pooled row.

Prompt mean is unweighted. Layer ``l`` is the output of decoder block ``l``.
The lens target is block ``L-1``.
"""

from __future__ import annotations

import torch
from torch import nn

from s2.lrp import lrp_mode

SKIP_FIRST = 16
MIN_EXPERT_COUNT = 50


def kept_positions(seq_len: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    if seq_len <= skip_first + 1:
        raise ValueError(f"seq_len {seq_len} leaves no kept positions")
    return torch.arange(skip_first, seq_len - 1)


def restricted_rows(
    model: nn.Module,
    input_ids: torch.Tensor,
    cotangents: torch.Tensor,
    *,
    arch: str | None,
    rows_per_pass: int = 8,
) -> torch.Tensor:
    """Return ``[n_rows, n_layers-1, d]`` fp32 source-mean rows for one prompt.

    ``cotangents`` is ``[n_rows, d]`` at the final block. ``arch`` selects LRP;
    ``None`` is plain gradient.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("one prompt")
    seq_len = input_ids.shape[1]
    kept = kept_positions(seq_len).to(input_ids.device)
    n_rows, d_model = cotangents.shape
    n_layers = len(model.model.layers)
    rows = torch.zeros(n_rows, n_layers - 1, d_model, dtype=torch.float32)
    context = lrp_mode(model, arch) if arch else _null_context()
    with context, torch.enable_grad():
        for start in range(0, n_rows, rows_per_pass):
            batch = cotangents[start : start + rows_per_pass]
            b = batch.shape[0]
            replicated = input_ids.expand(b, -1)
            captured: dict[int, torch.Tensor] = {}

            def make(index: int):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if index == 0:
                        hidden.requires_grad_(True)
                    captured[index] = hidden
                return hook

            handles = [model.model.layers[i].register_forward_hook(make(i)) for i in range(n_layers)]
            try:
                model(input_ids=replicated)
            finally:
                for handle in handles:
                    handle.remove()
            target = captured[n_layers - 1]
            sources = [captured[i] for i in range(n_layers - 1)]
            cot = torch.zeros_like(target)
            cot[:, kept, :] = batch.to(dtype=target.dtype)[:, None, :]
            grads = torch.autograd.grad(target, sources, cot, retain_graph=False)
            for layer, grad in enumerate(grads):
                rows[start : start + b, layer] = grad[:, kept, :].float().mean(dim=1).cpu()
            del grads, captured
    return rows


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
