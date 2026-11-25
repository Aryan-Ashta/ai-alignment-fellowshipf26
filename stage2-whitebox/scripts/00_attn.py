"""Gate 0.4. Which MPS attention path matches CPU fp32 SDPA at long context.

Reference is one CPU fp32 ``sdpa`` forward. Candidate MPS bf16 paths, in order:
(a) sdpa single forward, (b) sdpa chunked prefill, (c) eager chunked prefill.
The first path that passes at 8192 is written to ``configs/gen.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2.runio import require_ac_power, thermal_note, write_manifest  # noqa: E402

LENGTHS = (4096, 8192, 16384)
CHUNK = 1024
COSINE_MIN = 0.995
ARGMAX_MIN = 0.99
SCORE_SKIP_BYTES = 24 * (1024**3)
REF_DIR = ROOT / "scratch" / "attn_ref"


def load_model(repo: str, revision: str, dtype, device: str, attn: str):
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        dtype=dtype,
        attn_implementation=attn,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def wiki_ids(tok, n: int) -> torch.Tensor:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    parts: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if text:
            parts.append(text)
        blob = "\n".join(parts)
        ids = tok.encode(blob, add_special_tokens=False)
        if len(ids) >= n:
            return torch.tensor(ids[:n], dtype=torch.long)[None, :]
    raise SystemExit(f"WikiText stream produced only {len(ids)} tokens, need {n}")


def capture_last_block(model, input_ids, past=None):
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden.detach()

    handle = model.model.layers[-1].register_forward_hook(hook)
    try:
        kwargs = {"input_ids": input_ids, "use_cache": past is not None}
        if past is not None:
            kwargs["past_key_values"] = past
        out = model(**kwargs)
    finally:
        handle.remove()
    return captured["h"], out.logits.detach()


def chunked(model, input_ids, chunk: int):
    cache = DynamicCache(config=model.config)
    hiddens = []
    logits = []
    seq = input_ids.shape[1]
    for start in range(0, seq, chunk):
        piece = input_ids[:, start : start + chunk]
        hidden, logit = capture_last_block(model, piece, past=cache)
        hiddens.append(hidden.cpu())
        logits.append(logit.cpu())
    return torch.cat(hiddens, dim=1), torch.cat(logits, dim=1)


def single(model, input_ids):
    hidden, logits = capture_last_block(model, input_ids, past=None)
    return hidden.cpu(), logits.cpu()


def score_bytes(seq: int, heads: int, dtype_bytes: int) -> int:
    return heads * seq * seq * dtype_bytes


def compare(ref_h, ref_argmax, cand_h, cand_argmax) -> tuple[float, float, bool]:
    ref = ref_h[0].float()
    cand = cand_h[0].float()
    cos = F.cosine_similarity(ref, cand, dim=-1)
    min_cos = float(cos.min())
    agree = float((ref_argmax == cand_argmax).float().mean())
    ok = bool((cos >= COSINE_MIN).all().item()) and agree >= ARGMAX_MIN
    return min_cos, agree, ok


def save_ref(length: int, hidden, logits) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": hidden, "logits": logits.argmax(-1)}, REF_DIR / f"cpu_fp32_{length}.pt")


def load_ref(length: int):
    path = REF_DIR / f"cpu_fp32_{length}.pt"
    if not path.is_file():
        return None
    blob = torch.load(path, weights_only=True)
    return blob["hidden"], blob["logits"]


def main() -> None:
    require_ac_power()
    before = thermal_note()
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = models["olmo3_7b"]["dpo"]
    repo, revision = entry["repo"], entry["revision"]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    ids = wiki_ids(tok, max(LENGTHS))
    print(f"wiki tokens {ids.shape[1]}", flush=True)

    missing = [n for n in LENGTHS if load_ref(n) is None]
    if missing:
        print(f"CPU fp32 reference for {missing}", flush=True)
        cpu = load_model(repo, revision, torch.float32, "cpu", "sdpa")
        for n in missing:
            print(f"  cpu forward {n}", flush=True)
            hidden, logits = single(cpu, ids[:, :n])
            save_ref(n, hidden, logits)
        del cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("loading MPS bf16", flush=True)
    mps = load_model(repo, revision, torch.bfloat16, "mps", "sdpa")
    heads = mps.config.num_attention_heads
    results = {"a": {}, "b": {}, "c": {}}
    lines = [
        "# Gate 0.4 MPS attention",
        "",
        "Reference: CPU fp32 `attn_implementation='sdpa'`.",
        f"Pass: every final-block-output cosine ≥ {COSINE_MIN} and next-token argmax agreement ≥ {ARGMAX_MIN:.0%}.",
        f"Chunk size {CHUNK}.",
        "",
    ]
    eager_skip = []
    for n in LENGTHS:
        estimate = score_bytes(n, heads, 2)
        if estimate > SCORE_SKIP_BYTES:
            eager_skip.append(f"{n}: eager single forward skipped, score bytes {estimate}")
        else:
            eager_skip.append(f"{n}: eager single forward not in the candidate list; score bytes {estimate} ≤ 24 GB")
    lines.append("Eager single-forward skip log:")
    lines.extend(f"- {row}" for row in eager_skip)
    lines.append("")

    def run_path(label, model, fn):
        for n in LENGTHS:
            print(f"path {label} length {n}", flush=True)
            ref_h, ref_arg = load_ref(n)
            try:
                device = next(model.parameters()).device
                hidden, logits = fn(model, ids[:, :n].to(device))
                if device.type == "mps":
                    torch.mps.synchronize()
            except RuntimeError as exc:
                results[label][n] = {"ok": False, "error": str(exc)}
                lines.append(f"- path {label} {n}: ERROR {exc}")
                break
            cand_arg = logits.argmax(-1)
            min_cos, agree, ok = compare(ref_h, ref_arg, hidden, cand_arg)
            results[label][n] = {"ok": ok, "min_cos": min_cos, "agree": agree}
            lines.append(
                f"- path {label} {n}: min_cos={min_cos:.5f} argmax={agree:.4f} pass={ok}"
            )
            if not ok:
                break

    run_path("a", mps, lambda model, batch: single(model, batch))
    run_path("b", mps, lambda model, batch: chunked(model, batch, CHUNK))
    del mps
    print("reloading eager", flush=True)
    eager = load_model(repo, revision, torch.bfloat16, "mps", "eager")
    run_path("c", eager, lambda model, batch: chunked(model, batch, CHUNK))

    chosen = None
    for label in ("a", "b", "c"):
        if results[label].get(8192, {}).get("ok"):
            chosen = label
            break
    lines.append("")
    if chosen is None:
        lines.append("STOP: no path passed at 8192.")
    else:
        lines.append(f"Chosen path `{chosen}` (first that passed at 8192).")
        gen_path = ROOT / "configs" / "gen.yaml"
        gen = {}
        if gen_path.is_file():
            gen = yaml.safe_load(gen_path.read_text()) or {}
        gen["long_seq_attn"] = {"path": chosen, "chunk": CHUNK}
        gen.setdefault("astra_instruction", None)
        gen_path.write_text(yaml.safe_dump(gen, sort_keys=False))
        lines.append(f"Wrote `{gen_path.name}` long_seq_attn.path={chosen}.")

    del eager
    spot = load_model(repo, revision, torch.bfloat16, "mps", "eager" if chosen == "c" else "sdpa")
    spot_ids = ids[:, :512].to(next(spot.parameters()).device)
    if chosen in ("b", "c"):
        spot_h, spot_logits = chunked(spot, spot_ids, CHUNK)
    else:
        spot_h, spot_logits = single(spot, spot_ids)
    spot_arg = spot_logits.argmax(-1).cpu()
    spot_h = spot_h.cpu()
    del spot
    cpu512 = load_model(repo, revision, torch.float32, "cpu", "sdpa")
    ref_h, ref_logits = single(cpu512, ids[:, :512])
    min_cos, agree, ok = compare(ref_h, ref_logits.argmax(-1), spot_h, spot_arg)
    lines.append(
        f"512-token spot check path {chosen or 'a'}: min_cos={min_cos:.5f} argmax={agree:.4f} pass={ok}"
    )

    out = ROOT / "logs" / "preflight" / "attn.md"
    out.write_text("\n".join(lines) + "\n")
    write_manifest(
        ROOT / "logs" / "preflight",
        name="manifest_00_attn.json",
        script="00_attn.py",
        model_id=repo,
        revision=revision,
        thermal_note={"before": before, "after": thermal_note()},
    )
    print(f"wrote {out}")
    if chosen is None or not ok:
        raise SystemExit("Gate 0.4 failed; see logs/preflight/attn.md")


if __name__ == "__main__":
    main()
