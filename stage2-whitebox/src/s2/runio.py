"""Manifests, atomic saves, and Mac power/thermal checks for long jobs.

SPEC §13 fields are always written. uv has no ``pyproject.lock``; the lock
hash is the sha256 of ``uv.lock`` (approved plan).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

REFUSAL = "Long job refused: not on AC power"


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "uv.lock").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("uv.lock not found above src/s2/runio.py")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reports_ac_power(pmset_batt_text: str) -> bool:
    """True when ``pmset -g batt`` says the machine is drawing AC power."""
    return "AC Power" in pmset_batt_text


def pmset_batt() -> str:
    return subprocess.check_output(["pmset", "-g", "batt"], text=True)


def require_ac_power(text: str | None = None) -> None:
    """Raise unless ``pmset -g batt`` reports AC power.

    ``text`` is for tests. Production callers omit it and read ``pmset``.
    A non-macOS host has no ``pmset``; the rented GPU box is not gated here.
    """
    if text is None and platform.system() != "Darwin":
        return
    if text is None:
        text = pmset_batt()
    if not reports_ac_power(text):
        raise RuntimeError(REFUSAL)


def thermal_note() -> str:
    return subprocess.check_output(["pmset", "-g", "therm"], text=True)


def code_commit() -> str:
    root = project_root()
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def device_name() -> str:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    if torch is not None and torch.backends.mps.is_available():
        return f"{model} mps"
    return model or platform.machine()


def manifest_fields(**extra: Any) -> dict[str, Any]:
    """Every SPEC §13 field. ``extra`` fills run-specific values and may add keys."""
    root = project_root()
    lock = root / "uv.lock"
    payload: dict[str, Any] = {
        "code_commit": code_commit(),
        "pyproject_lock_hash": sha256_file(lock),
        "lock_file": "uv.lock",
        "versions": {
            "python": platform.python_version(),
            "torch": _module_version("torch"),
            "transformers": _module_version("transformers"),
            "mlx": _module_version("mlx"),
            "mlx_lm": _module_version("mlx_lm"),
        },
        "os": platform.platform(),
        "device_name": extra.get("device_name") or device_name(),
        "model_id": extra.get("model_id"),
        "revision": extra.get("revision"),
        "item_file_sha256": extra.get("item_file_sha256"),
        "fitting_corpus_sha256": extra.get("fitting_corpus_sha256"),
        "seeds": extra.get("seeds"),
        "wall_time_s": extra.get("wall_time_s"),
        "peak_memory": extra.get("peak_memory"),
        "thermal_note": extra.get("thermal_note"),
    }
    for key, value in extra.items():
        if key not in payload:
            payload[key] = value
    return payload


def write_manifest(directory: str | Path, *, name: str = "manifest.json", **extra: Any) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    payload = manifest_fields(**extra)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
