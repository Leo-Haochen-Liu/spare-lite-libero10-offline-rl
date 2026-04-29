#!/usr/bin/env python3
"""Generate LIBERO-10 per-task comparison figures from parsed rollout counts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COUNTS_PATH = ROOT / "analysis" / "libero10_final_comparison_counts.json"
FIGURES_DIR = ROOT / "figures"

TASKS = [f"T{i}" for i in range(10)]
TASK_MAPPING = {
    "T0": "soup + tomato sauce -> basket",
    "T1": "cream cheese + butter -> basket",
    "T2": "turn on stove + put moka pot",
    "T3": "black bowl -> bottom drawer and close",
    "T4": "white mug left plate + yellow/white mug right plate",
    "T5": "book -> back compartment of caddy",
    "T6": "white mug on plate + pudding right of plate",
    "T7": "soup + cream cheese -> basket",
    "T8": "both moka pots -> stove",
    "T9": "yellow/white mug -> microwave and close",
}

COLORS = {
    "official_sft": "#4f79b8",
    "official_postrl": "#6faa61",
    "sft_full409_r1": "#6faa61",
    "sft_full409_r1r2": "#e19a43",
    "postrl_rlbase_r1": "#e19a43",
    "postrl_rlbase_r1r2": "#df7f4f",
}


def load_counts() -> dict:
    return json.loads(COUNTS_PATH.read_text())


def pct(entry: dict, task: str) -> float:
    return 100.0 * entry["per_task"][task]["rate"]


def avg(entry: dict) -> float:
    return 100.0 * entry["success"] / entry["total"]


def draw_mapping(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(0.0, 0.94, "Task mapping", fontsize=15, fontweight="bold", va="top")
    left = TASKS[:5]
    right = TASKS[5:]
    y0 = 0.76
    dy = 0.16
    for i, task in enumerate(left):
        ax.text(0.0, y0 - i * dy, f"{task}: {TASK_MAPPING[task]}", fontsize=12, va="top")
    for i, task in enumerate(right):
        ax.text(0.53, y0 - i * dy, f"{task}: {TASK_MAPPING[task]}", fontsize=12, va="top")


def plot_group(filename: str, title: str, series: list[tuple[str, str]], footnote: str) -> None:
    counts = load_counts()
    x = np.arange(len(TASKS))
    width = min(0.8 / len(series), 0.23)

    fig = plt.figure(figsize=(18, 11), dpi=180)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.8, 1.55], hspace=0.22)
    ax = fig.add_subplot(grid[0])
    mapping_ax = fig.add_subplot(grid[1])

    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    for idx, (key, label) in enumerate(series):
        entry = counts[key]
        values = [pct(entry, task) for task in TASKS]
        legend_label = f"{label} ({entry['total']} rollouts, avg {avg(entry):.1f}%)"
        bars = ax.bar(x + offsets[idx], values, width, label=legend_label, color=COLORS[key], alpha=0.96)
        for bar, value in zip(bars, values):
            label_y = value + 1.5 if value > 0 else 1.2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#202020",
            )

    ax.axhline(100, color="#8a8a8a", linestyle="--", linewidth=1.1)
    ax.text(len(TASKS) - 0.08, 101.2, "100%", ha="right", va="bottom", color="#777", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.8, len(TASKS) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.grid(axis="y", alpha=0.22)
    ax.set_title(title, fontsize=20, pad=34)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.095),
        ncol=2 if len(series) <= 3 else 2,
        frameon=False,
        fontsize=11.5,
        columnspacing=1.8,
        handlelength=1.6,
    )

    draw_mapping(mapping_ax)
    mapping_ax.text(0.0, -0.05, footnote, fontsize=11.5, color="#333", va="top")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_group(
        "libero10_sft_baseline_full409_offline_rl_per_task.png",
        "LIBERO-10: official SFT baseline vs offline RL from SFT",
        [
            ("official_sft", "official SFT"),
            ("sft_full409_r1", "R1 from SFT"),
            ("sft_full409_r1r2", "R1+R2 from SFT"),
        ],
        "Strict controlled comparison using full LIBERO-10 expert transitions plus 409 real SFT failure episodes.",
    )
    plot_group(
        "libero10_postrl_baseline_offline_rl_per_task.png",
        "LIBERO-10: official RL baseline vs offline RL from RL checkpoint",
        [
            ("official_sft", "official SFT"),
            ("official_postrl", "official post-RL"),
            ("postrl_rlbase_r1", "R1 from post-RL"),
            ("postrl_rlbase_r1r2", "R1+R2 from post-RL"),
        ],
        "Earlier RL-baseline offline checkpoints were trained from the spatial-offline data source and are included as a reference comparison.",
    )


if __name__ == "__main__":
    main()
