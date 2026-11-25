import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

from s2.models import LoadedModel


def _tiny() -> LoadedModel:
    config = Olmo3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        layer_types=["full_attention"] * 4,
        sliding_window=8,
        max_position_embeddings=32,
        eos_token_id=0,
    )
    hf = Olmo3ForCausalLM(config).float().eval()
    hf.requires_grad_(False)
    return LoadedModel(
        hf=hf,
        tok=None,
        layers=hf.model.layers,
        n_layers=4,
        d_model=64,
        final_norm=hf.model.norm,
        lm_head=hf.lm_head,
        number_ids=list(range(100)),
        key="tiny",
        revision="0" * 40,
    )


def test_capture_and_identity_replace_do_not_change_logits():
    model = _tiny()
    ids = torch.randint(0, 128, (1, 8))
    bare = model.hf(input_ids=ids).logits
    with model.capture([0, 3]) as stored:
        captured = model.hf(input_ids=ids).logits
    assert torch.equal(bare, captured)
    assert set(stored) == {0, 3}
    with model.replace({1: lambda h: h}):
        replaced = model.hf(input_ids=ids).logits
    assert torch.equal(bare, replaced)
    with model.capture([model.n_layers - 1]) as stored:
        logits = model.hf(input_ids=ids).logits
    readout = model.final_readout(stored[model.n_layers - 1])
    assert torch.equal(readout, logits)


def test_capture_with_grad_roots_the_graph():
    model = _tiny()
    ids = torch.randint(0, 128, (1, 6))
    with model.capture_with_grad(0) as stored:
        logits = model.hf(input_ids=ids).logits
        assert stored[0].requires_grad
        (grad,) = torch.autograd.grad(logits.sum(), stored[0])
    assert grad.shape == stored[0].shape
    assert torch.isfinite(grad).all()
