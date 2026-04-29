#!/usr/bin/env python3
"""Plot offline-RL training diagnostics from saved IQL-style train logs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "training_logs"
FIG_DIR = ROOT / "figures"

METRIC_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?)")

RUNS = {
    "sft_full409": {
        "title": "Training diagnostics: offline RL from official SFT",
        "subtitle": "LIBERO-10 full expert demonstrations + 409 real SFT failure episodes",
        "files": {
            "R1 (lambda=0.0)": LOG_DIR / "sft_full409_r1_train.log",
            "R1+R2 (lambda=0.2)": LOG_DIR / "sft_full409_r1r2_train.log",
        },
        "output": FIG_DIR / "libero10_sft_full409_training_diagnostics.png",
    },
    "postrl_full409": {
        "title": "Training diagnostics: offline RL from official post-RL",
        "subtitle": "Same LIBERO-10 full409 data and hyperparameters, stronger starting policy",
        "files": {
            "R1 (lambda=0.0)": LOG_DIR / "postrl_full409_r1_train.log",
            "R1+R2 (lambda=0.2)": LOG_DIR / "postrl_full409_r1r2_train.log",
        },
        "output": FIG_DIR / "libero10_postrl_full409_training_diagnostics.png",
    },
}


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if "[iql-style]" not in line or " step=" not in line:
            continue
        metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if "step" in metrics:
            metrics["step"] = int(metrics["step"])
            rows.append(metrics)
    if not rows:
        raise ValueError(f"No step metrics parsed from {path}")
    return pd.DataFrame(rows).sort_values("step")


def smooth(series: pd.Series, window: int = 15) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_run(config: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames = {label: parse_log(path) for label, path in config["files"].items()}

    colors = {
        "R1 (lambda=0.0)": "#6faa61",
        "R1+R2 (lambda=0.2)": "#e6862e",
    }
    metrics = [
        ("reward", "Batch shaped reward"),
        ("loss", "Total loss"),
        ("actor_loss", "Actor loss"),
        ("q_loss", "Q loss"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), dpi=180)
    axes = axes.ravel()
    fig.suptitle(config["title"], fontsize=18, y=0.975)
    fig.text(0.5, 0.935, config["subtitle"], ha="center", fontsize=11, color="#555555")

    for ax, (metric, ylabel) in zip(axes, metrics):
        for label, df in frames.items():
            ax.plot(
                df["step"],
                smooth(df[metric]),
                label=label,
                color=colors[label],
                linewidth=2.2,
            )
            ax.plot(
                df["step"],
                df[metric],
                color=colors[label],
                alpha=0.14,
                linewidth=0.8,
            )
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False)
    fig.tight_layout(rect=(0.035, 0.035, 0.98, 0.875))
    fig.savefig(config["output"], bbox_inches="tight")
    print(config["output"])


def main() -> None:
    for config in RUNS.values():
        plot_run(config)


if __name__ == "__main__":
    main()
