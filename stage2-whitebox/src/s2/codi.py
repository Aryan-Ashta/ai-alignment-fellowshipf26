"""CODI loader without peft. Merges LoRA by hand and runs the latent loop.

Reference: ``cywinski/codi`` ``src/model.py`` and
``experiments/3_logit_lens_latents.py``. LoRA scale is alpha/r = 32/128.
If a base weight is absent from the bin, the gated Meta checkpoint is
required and this module stops rather than inventing weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
LORA_SCALE = 32 / 128
PREFIXES = (
    "codi.base_model.model.",
    "base_model.model.",
    "codi.",
)


def _entry(key: str) -> dict:
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    return models[key]


def _strip(key: str) -> str:
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_codi_bin() -> dict[str, torch.Tensor]:
    entry = _entry("codi")
    path = Path(snapshot_download(entry["repo"], revision=entry["revision"]))
    weight = path / "pytorch_model.bin"
    if not weight.is_file():
        raise RuntimeError(f"missing {weight}")
    return torch.load(weight, map_location="cpu", weights_only=True)


def write_key_report(state: dict[str, torch.Tensor], path: Path) -> None:
    lines = ["# CODI state-dict keys", "", f"{len(state)} tensors.", ""]
    for key, value in sorted(state.items()):
        lines.append(f"- `{key}` {tuple(value.shape)} {value.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def merge_lora(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], nn.Sequential]:
    bases: dict[str, torch.Tensor] = {}
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    prj: dict[str, torch.Tensor] = {}
    other: dict[str, torch.Tensor] = {}
    for raw, tensor in state.items():
        key = _strip(raw)
        if ".lora_A." in key and key.endswith(".weight"):
            lora_a[key.split(".lora_A.")[0]] = tensor
        elif ".lora_B." in key and key.endswith(".weight"):
            lora_b[key.split(".lora_B.")[0]] = tensor
        elif key.endswith(".base_layer.weight"):
            bases[key[: -len(".base_layer.weight")] + ".weight"] = tensor
        elif key.startswith("prj."):
            prj[key[len("prj.") :]] = tensor
        elif "lora_" not in key:
            other[key] = tensor
    if not bases:
        raise SystemExit(
            "CODI bin has no base_layer weights. The gated Meta Llama weights are required. "
            "Stop: Aryan must accept the licence and export HF_TOKEN. No account will be created."
        )
    merged = dict(other)
    missing_pairs = []
    for name, base in bases.items():
        module = name[: -len(".weight")]
        if module not in lora_a or module not in lora_b:
            missing_pairs.append(module)
            merged[name] = base
            continue
        a = lora_a[module].float()
        b = lora_b[module].float()
        merged[name] = (base.float() + LORA_SCALE * (b @ a)).to(dtype=base.dtype)
    if missing_pairs:
        raise RuntimeError(f"LoRA pair missing for {missing_pairs[:8]}")
    projector = nn.Sequential(
        nn.Dropout(0.0),
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
    )
    projector.add_module("ln", nn.LayerNorm(2048))
    missing, unexpected = projector.load_state_dict(prj, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"prj load missing={missing} unexpected={unexpected}")
    return merged, projector


def build_merged(device: str = "cpu", dtype=torch.bfloat16) -> tuple[LlamaForCausalLM, nn.Sequential, object]:
    codi = _entry("codi")
    base = _entry("codi_base")
    snapshot_download(base["repo"], revision=base["revision"], allow_patterns=["config.json", "generation_config.json", "tokenizer*", "special_tokens*"])
    tok = AutoTokenizer.from_pretrained(codi["repo"], revision=codi["revision"])
    config = LlamaConfig.from_pretrained(base["repo"], revision=base["revision"])
    model = LlamaForCausalLM(config).to(dtype=dtype)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    state = load_codi_bin()
    write_key_report(state, ROOT / "logs" / "preflight" / "codi_keys.md")
    merged, projector = merge_lora(state)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    base_missing = [k for k in missing if not k.startswith("lm_head")]
    if base_missing or unexpected:
        raise RuntimeError(f"base load missing={base_missing[:12]} unexpected={unexpected[:12]}")
    if "lm_head.weight" not in merged:
        model.tie_weights()
    model.to(device)
    projector.to(device=device, dtype=dtype)
    model.eval()
    projector.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in projector.parameters():
        param.requires_grad_(False)
    return model, projector, tok


def codi_latent_run(model, projector, tok, prompt: str, n_latent: int = 6, patch: dict[int, torch.Tensor] | None = None) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    bocot = tok.convert_tokens_to_ids("<|bocot|>")
    eocot = tok.convert_tokens_to_ids("<|eocot|>")
    bot = torch.tensor([[tok.eos_token_id, bocot]], device=device)
    input_ids = torch.cat([ids, bot], dim=1)
    hidden: dict[int, list[torch.Tensor]] = {i: [] for i in range(model.config.num_hidden_layers)}
    latent_inputs = []

    def capture(output):
        states = output.hidden_states
        # hidden_states[0] is embeddings; [1:] are block outputs.
        for layer, tensor in enumerate(states[1:]):
            hidden[layer].append(tensor[:, -1].detach().cpu())

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        capture(out)
        latent = out.hidden_states[-1][:, -1]
        latent = projector(latent.unsqueeze(1)).to(dtype)
        latent_inputs.append(latent.detach().cpu())
        cache = out.past_key_values
        for step in range(n_latent):
            if patch and step in patch:
                latent = patch[step].to(device=device, dtype=dtype)
                if latent.ndim == 1:
                    latent = latent.view(1, 1, -1)
            out = model(inputs_embeds=latent, use_cache=True, output_hidden_states=True, past_key_values=cache)
            capture(out)
            cache = out.past_key_values
            latent = projector(out.hidden_states[-1][:, -1].unsqueeze(1)).to(dtype)
            latent_inputs.append(latent.detach().cpu())
        eocot_emb = model.model.embed_tokens(torch.tensor([[eocot]], device=device))
        out = model(inputs_embeds=eocot_emb, use_cache=True, past_key_values=cache)
        logits = out.logits[:, -1, : model.config.vocab_size - 1]
        generated = [int(logits.argmax(-1))]
        cache = out.past_key_values
        for _ in range(15):
            emb = model.model.embed_tokens(torch.tensor([[generated[-1]]], device=device))
            out = model(inputs_embeds=emb, use_cache=True, past_key_values=cache)
            cache = out.past_key_values
            nxt = int(out.logits[:, -1, : model.config.vocab_size - 1].argmax(-1))
            generated.append(nxt)
            if nxt == tok.eos_token_id:
                break
    text = tok.decode(generated, skip_special_tokens=True)
    return {
        "answer_text": text,
        "answer_ids": generated,
        "latent_inputs": latent_inputs,
        "hidden": hidden,
    }
