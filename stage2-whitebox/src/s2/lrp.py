"""LRP backward rules for the restricted J++ / R-lens fit.

Olmo3 coverage matches jpp ``rlens``: LN rule on every RMSNorm outside
``self_attn`` (so ``q_norm`` / ``k_norm`` stay stock), identity rule
``z * sigmoid(z).detach()`` and half rule ``0.5*p + 0.5*p.detach()`` on
``Olmo3MLP``. Forward values copy jpp's ``_olmo3_rmsnorm_lrp_forward``
(weight applied in fp32 before the cast).

Llama has no rules in jpp. The walker semantics are the same ones jpp uses
(every RMSNorm outside ``self_attn``, every gated MLP), applied to Llama
classes. The Llama norm forward matches stock ``LlamaRMSNorm``, which
multiplies the weight after the downcast.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
from torch import Tensor, nn

_MISSING = object()
_QK_NORM_NAMES = ("q_norm", "k_norm")


def _olmo3_rmsnorm_lrp_forward(self: Any, hidden_states: Tensor) -> Tensor:
    input_dtype = hidden_states.dtype
    x_f = hidden_states.to(torch.float32)
    scale = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    result: Tensor = (self.weight * (x_f * scale)).to(input_dtype)
    return result


def _llama_rmsnorm_lrp_forward(self: Any, hidden_states: Tensor) -> Tensor:
    input_dtype = hidden_states.dtype
    x_f = hidden_states.to(torch.float32)
    scale = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    return self.weight * (x_f * scale).to(input_dtype)


def _silu_with_detached_sigmoid(z: Tensor) -> Tensor:
    exact = torch.nn.functional.silu(z)
    identity = z * torch.sigmoid(z).detach()
    return identity + (exact - identity).detach()


def _halve_gradient(x: Tensor) -> Tensor:
    halved = 0.5 * x + 0.5 * x.detach()
    return halved + (x - halved).detach()


def _gated_mlp_lrp_forward(self: Any, x: Tensor) -> Tensor:
    product = _silu_with_detached_sigmoid(self.gate_proj(x)) * self.up_proj(x)
    return self.down_proj(_halve_gradient(product))


_NORM_FORWARDS = {
    "Olmo3RMSNorm": _olmo3_rmsnorm_lrp_forward,
    "LlamaRMSNorm": _llama_rmsnorm_lrp_forward,
}
_MLP_CLASSES = {"Olmo3MLP": "olmo3", "LlamaMLP": "llama"}


def _is_qk_or_attention_norm(qualified_name: str) -> bool:
    parts = qualified_name.split(".")
    return parts[-1] in _QK_NORM_NAMES or "self_attn" in parts


def _patch(saved: list[tuple[nn.Module, object]], module: nn.Module, forward) -> None:
    saved.append((module, module.__dict__.get("forward", _MISSING)))
    object.__setattr__(module, "forward", MethodType(forward, module))


def _restore(saved: list[tuple[nn.Module, object]]) -> None:
    for module, original in reversed(saved):
        if original is _MISSING:
            module.__dict__.pop("forward", None)
        else:
            object.__setattr__(module, "forward", original)


class LrpReport:
    def __init__(self) -> None:
        self.ln_rule: list[str] = []
        self.mlp_rule: list[str] = []

    def names(self) -> dict[str, list[str]]:
        return {"ln_rule": list(self.ln_rule), "mlp_rule": list(self.mlp_rule)}


@contextmanager
def lrp_mode(model: nn.Module, arch: str) -> Iterator[LrpReport]:
    """Patch instance forwards for ``arch`` in {"olmo3", "llama"} and restore them."""
    if arch not in {"olmo3", "llama"}:
        raise ValueError(f"arch must be 'olmo3' or 'llama', got {arch!r}")
    saved: list[tuple[nn.Module, object]] = []
    report = LrpReport()
    try:
        for name, module in model.named_modules():
            cls = type(module).__name__
            if cls in _NORM_FORWARDS and arch == ("olmo3" if cls.startswith("Olmo") else "llama"):
                if _is_qk_or_attention_norm(name):
                    continue
                _patch(saved, module, _NORM_FORWARDS[cls])
                report.ln_rule.append(name)
            elif cls in _MLP_CLASSES and _MLP_CLASSES[cls] == arch:
                _patch(saved, module, _gated_mlp_lrp_forward)
                report.mlp_rule.append(name)
        if not report.ln_rule or not report.mlp_rule:
            raise RuntimeError(f"LRP patch missed a family on {type(model).__name__}: {report.names()}")
        yield report
    finally:
        _restore(saved)
