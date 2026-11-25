"""One figure per question. Lens colours are fixed: J++ C0, R-lens C1, logit C2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LENS_COLOURS = {"jpp": "C0", "rlens": "C1", "logit": "C2", "null": "0.6"}


def plot_depth(csv_path: Path, out: Path) -> None:
    table = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        table["rl_step"],
        table["depth"],
        yerr=[table["depth"] - table["ci_low"], table["ci_high"] - table["depth"]],
        fmt="o",
        color=LENS_COLOURS["jpp"],
        capsize=3,
    )
    ax.set_xlabel("RL step")
    ax.set_ylabel("internal step depth")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_swap(csv_path: Path, out: Path) -> None:
    table = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, block in table.groupby("variant"):
        ax.plot(block["layer"], block["success"], label=str(variant))
    ax.set_xlabel("layer")
    ax.set_ylabel("swap success")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
