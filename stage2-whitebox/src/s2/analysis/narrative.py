"""Draft narrative. Only lines with a measured value are included."""

from __future__ import annotations

from pathlib import Path


def write_narrative(measured: dict[str, str], out: Path) -> str:
    lines = [
        "# Stage 2 results draft",
        "",
        "Readability below is among the 100 number tokens only.",
        "Pre-registered comparisons are the depth trend across RL steps and the no-CoT accuracy trend.",
        "Everything else is exploratory until Aryan freezes it.",
        "",
    ]
    if not measured:
        lines.append("No measured comparison is available yet.")
    else:
        lines.extend(f"- {key}: {value}" for key, value in measured.items())
    lines.append("")
    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text
