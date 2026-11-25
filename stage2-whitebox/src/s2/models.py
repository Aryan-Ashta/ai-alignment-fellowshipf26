"""Checkpoint loading and residual-stream hooks.

Layer ℓ is the residual leaving decoder block ℓ (``layers[ℓ]`` output, or
``output[0]`` when that output is a tuple). The lens target is layer ``L-1``.
``encode_fit`` copies ``jlens.hf.HFLensModel.encode``: ``add_bos_token`` is
set when the tokenizer has a BOS id, then the text is truncated to 128 tokens.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import yaml
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def _block_output(output):
    return output if torch.is_tensor(output) else output[0]


def _replace_output(output, edited):
    if torch.is_tensor(output):
        return edited
    return (edited, *output[1:])


@dataclass
class LoadedModel:
    hf: nn.Module
    tok: object
    layers: nn.ModuleList
    n_layers: int
    d_model: int
    final_norm: nn.Module
    lm_head: nn.Linear
    number_ids: list[int]
    key: str
    revision: str

    def final_readout(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden))

    def encode_fit(self, text: str, max_length: int = 128) -> torch.Tensor:
        if getattr(self.tok, "bos_token_id", None) is not None and hasattr(self.tok, "add_bos_token"):
            self.tok.add_bos_token = True
        encoded = self.tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        device = next(self.hf.parameters()).device
        return encoded.input_ids.to(device)

    @contextmanager
    def capture(self, layers: list[int], positions: list[int] | None = None) -> Iterator[dict[int, torch.Tensor]]:
        stored: dict[int, torch.Tensor] = {}

        def make(index: int):
            def hook(_module, _inputs, output):
                hidden = _block_output(output).detach()
                stored[index] = hidden if positions is None else hidden[:, positions, :]
            return hook

        handles = [self.layers[i].register_forward_hook(make(i)) for i in layers]
        try:
            yield stored
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def replace(
        self,
        editors: dict[int, Callable[[torch.Tensor], torch.Tensor]],
        positions: list[int] | None = None,
    ) -> Iterator[None]:
        def make(index: int, fn: Callable[[torch.Tensor], torch.Tensor]):
            def hook(_module, _inputs, output):
                hidden = _block_output(output)
                edited = hidden.clone()
                if positions is None:
                    updated = fn(hidden.float()).to(dtype=hidden.dtype)
                    return _replace_output(output, updated)
                selected = hidden[:, positions, :].float()
                edited[:, positions, :] = fn(selected).to(dtype=hidden.dtype)
                return _replace_output(output, edited)
            return hook

        handles = [
            self.layers[index].register_forward_hook(make(index, fn))
            for index, fn in editors.items()
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def capture_with_grad(self, min_layer: int, layers: list[int] | None = None) -> Iterator[dict[int, torch.Tensor]]:
        wanted = list(range(self.n_layers) if layers is None else layers)
        if min_layer not in wanted:
            wanted = sorted({*wanted, min_layer})
        stored: dict[int, torch.Tensor] = {}

        def make(index: int):
            def hook(_module, _inputs, output):
                hidden = _block_output(output)
                if index == min_layer:
                    hidden.requires_grad_(True)
                stored[index] = hidden
                if index == min_layer and not torch.is_tensor(output):
                    return _replace_output(output, hidden)
            return hook

        handles = [self.layers[i].register_forward_hook(make(i)) for i in wanted]
        try:
            yield stored
        finally:
            for handle in handles:
                handle.remove()


def _number_ids(family: str = "olmo3") -> list[int]:
    path = ROOT / "logs" / "preflight" / f"number_token_ids_{family}.json"
    payload = json.loads(path.read_text())
    return [payload["number_token_ids"][str(n)] for n in range(100)]


def load(model_key: str, device: str = "mps", dtype=torch.bfloat16, attn: str = "sdpa") -> LoadedModel:
    """Load ``family.slot`` or a top-level key by the SHA in ``configs/models.yaml``."""
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    if "." in model_key:
        family, slot = model_key.split(".", 1)
        entry = models[family][slot]
    else:
        entry = models[model_key]
        family = "codi" if model_key.startswith("codi") else "olmo3"
    revision = entry["revision"]
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError(f"revision must be a full SHA, got {revision!r}")
    hf = AutoModelForCausalLM.from_pretrained(
        entry["repo"],
        revision=revision,
        dtype=dtype,
        attn_implementation=attn,
        low_cpu_mem_usage=True,
    )
    hf.to(device)
    hf.eval()
    hf.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(entry["repo"], revision=revision)
    return LoadedModel(
        hf=hf,
        tok=tok,
        layers=hf.model.layers,
        n_layers=hf.config.num_hidden_layers,
        d_model=hf.config.hidden_size,
        final_norm=hf.model.norm,
        lm_head=hf.lm_head,
        number_ids=_number_ids("olmo3" if family.startswith("olmo") else "codi"),
        key=model_key,
        revision=revision,
    )
