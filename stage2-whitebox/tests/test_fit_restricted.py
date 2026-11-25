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


def test_sparse_expert_falls_back_to_pooled():
    from s2.fit_restricted import finalize_experts

    expert_sum = torch.tensor([[[2.0, 0.0]], [[0.0, 8.0]]])
    expert_count = torch.tensor([1.0, 3.0])
    experts, pooled, fallback = finalize_experts(expert_sum, expert_count, min_count=2)
    assert bool(fallback[0]) and not bool(fallback[1])
    assert torch.allclose(experts[0], pooled)
    assert torch.allclose(experts[1, 0], torch.tensor([0.0, 8.0 / 3]))


def test_all_at_once_cotangent_matches_per_position_sum():
    from s2.fit_restricted import kept_positions

    torch.manual_seed(2)
    model = _tiny()
    ids = torch.randint(0, 64, (1, 20))
    cotangent = torch.randn(1, 32)
    together = restricted_rows(model, ids, cotangent, arch=None, rows_per_pass=1)
    kept = kept_positions(ids.shape[1])
    acc = torch.zeros_like(together)
    for position in kept.tolist():
        partial = torch.zeros_like(cotangent)
        # Place the cotangent at one target by running the model hook path directly.
        captured = {}

        def make(index, captured=captured):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if index == 0:
                    hidden.requires_grad_(True)
                captured[index] = hidden
            return hook

        handles = [model.model.layers[i].register_forward_hook(make(i)) for i in range(3)]
        with torch.enable_grad():
            model(input_ids=ids)
            for handle in handles:
                handle.remove()
            target = captured[2]
            sources = [captured[0], captured[1]]
            cot = torch.zeros_like(target)
            cot[0, position] = cotangent[0].to(target.dtype)
            grads = torch.autograd.grad(target, sources, cot)
        for layer, grad in enumerate(grads):
            acc[0, layer] += grad[0, kept].float().sum(0).cpu()
        del captured
    alone = acc / len(kept)
    rel = (together - alone).norm() / together.norm().clamp_min(1e-12)
    assert float(rel) < 1e-5, float(rel)
