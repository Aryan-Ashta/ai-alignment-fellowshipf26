import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

from s2.fit_restricted import restricted_rows


def _tiny():
    config = Olmo3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        layer_types=["full_attention"] * 3,
        sliding_window=8,
        max_position_embeddings=40,
        eos_token_id=0,
    )
    model = Olmo3ForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


def test_batched_vjp_matches_per_row_loop():
    torch.manual_seed(0)
    model = _tiny()
    ids = torch.randint(0, 64, (1, 24))
    cotangents = torch.randn(4, 32)
    one = restricted_rows(model, ids, cotangents, arch=None, rows_per_pass=1)
    batched = restricted_rows(model, ids, cotangents, arch="olmo3", rows_per_pass=4)
    # Plain and LRP differ. Compare batching within plain mode.
    batched_plain = restricted_rows(model, ids, cotangents, arch=None, rows_per_pass=4)
    rel = (one - batched_plain).norm() / one.norm().clamp_min(1e-12)
    assert float(rel) < 1e-5, float(rel)
    assert batched.shape == one.shape


def test_resume_sum_matches_uninterrupted_run():
    from s2.fit_restricted import sum_prompt_rows

    torch.manual_seed(1)
    model = _tiny()
    prompts = [torch.randint(0, 64, (1, 24)) for _ in range(2)]
    cotangents = torch.randn(2, 32)
    full = sum_prompt_rows(model, prompts, cotangents, arch=None, rows_per_pass=2)
    first = sum_prompt_rows(model, prompts[:1], cotangents, arch=None, rows_per_pass=2)
    resumed = sum_prompt_rows(model, prompts, cotangents, arch=None, rows_per_pass=2, start=1, acc=first)
    assert torch.equal(full, resumed)
