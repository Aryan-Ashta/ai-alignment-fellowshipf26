import torch
from transformers import LlamaConfig, LlamaForCausalLM, Olmo3Config, Olmo3ForCausalLM

from s2.lrp import lrp_mode


def _olmo():
    config = Olmo3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=512,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=8,
        max_position_embeddings=32,
    )
    model = Olmo3ForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


def _llama():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=512,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


def _logits(model, ids):
    return model(input_ids=ids).logits


def test_lrp_forward_is_bit_identical():
    ids = torch.randint(0, 512, (2, 12))
    for arch, model in (("olmo3", _olmo()), ("llama", _llama())):
        off = _logits(model, ids)
        with lrp_mode(model, arch):
            on = _logits(model, ids)
        assert torch.equal(off, on), arch


def test_olmo_gradients_match_jpp_rlens():
    from workspace_lens.lrp.lrp import apply_lrp_rules_for_mode

    torch.manual_seed(0)
    model = _olmo()
    ids = torch.randint(0, 512, (1, 8))
    cot = torch.randn(1, 8, model.config.vocab_size)

    def backward(context):
        embeds = model.model.embed_tokens(ids).detach().requires_grad_(True)
        with context:
            logits = model(inputs_embeds=embeds).logits
        (grad,) = torch.autograd.grad(logits, embeds, cot)
        return grad

    ours = backward(lrp_mode(model, "olmo3"))
    theirs = backward(apply_lrp_rules_for_mode(model, "rlens"))
    rel = (ours - theirs).norm() / theirs.norm().clamp_min(1e-12)
    assert float(rel) < 1e-6, float(rel)


def test_olmo_patch_names_match_jpp():
    from workspace_lens.lrp.lrp import apply_lrp_rules_for_mode

    model = _olmo()
    with lrp_mode(model, "olmo3") as ours:
        our_names = ours.names()
    with apply_lrp_rules_for_mode(model, "rlens") as report:
        their_names = {"ln_rule": list(report.ln_rule), "mlp_rule": list(report.mlp_rule)}
    assert our_names == their_names
    assert any(name.endswith("q_norm") for name, _ in model.named_modules())
    assert not any("q_norm" in name or "k_norm" in name for name in our_names["ln_rule"])
