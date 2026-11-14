"""Static charts for the report. One fixed colour per model, never reassigned."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {"popularity": "#2a78d6", "matrix_factorization": "#eb6834", "hybrid": "#1baf7a"}
MODEL_LABELS = {"popularity": "Popularity", "matrix_factorization": "Matrix factorisation", "hybrid": "Hybrid"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def metric_bars(table: pd.DataFrame, k: int, path) -> None:
    metrics = [(f"precision_at_{k}", f"Precision@{k}"), (f"recall_at_{k}", f"Recall@{k}"),
               (f"ndcg_at_{k}", f"NDCG@{k}"), (f"coverage_at_{k}", f"Coverage@{k}"),
               (f"diversity_at_{k}", f"Diversity@{k}")]
    models = list(table.index)
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 2.8))
    for ax, (col, label) in zip(axes, metrics):
        vals = table[col].to_numpy()
        x = np.arange(len(models))
        ax.bar(x, vals, width=0.62, color=[MODEL_COLORS[m] for m in models], edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_ylim(0, max(vals) * 1.22)
        _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in models]
    fig.legend(handles, [MODEL_LABELS[m] for m in models], loc="lower center", ncol=len(models), frameon=False,
               fontsize=8)
    fig.suptitle(f"Test-period ranking quality, warm users, K={k}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def precision_recall_curves(curves: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, metric, label in ((axes[0], "precision", "Precision@K"), (axes[1], "recall", "Recall@K")):
        for model, grp in curves.groupby("model", sort=False):
            ax.plot(grp["k"], grp[metric], color=MODEL_COLORS[model], linewidth=2, label=MODEL_LABELS[model])
        ax.set_xlabel("K", fontsize=8, color=MUTED)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xticks([1, 5, 10, 15, 20])
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def popularity_profile(rec_pop_pct: dict[str, np.ndarray], path) -> None:
    """Share of each model's recommendations that come from the head, middle and tail of the catalogue."""
    buckets = [("Long tail\n(bottom 50%)", 0.0, 0.5), ("Middle\n(50-90%)", 0.5, 0.9), ("Head\n(top 10%)", 0.9, 1.01)]
    models = list(rec_pop_pct)
    fig, ax = plt.subplots(figsize=(7, 2.9))
    width = 0.26
    x = np.arange(len(buckets))
    for j, model in enumerate(models):
        pct = rec_pop_pct[model]
        shares = [((pct >= lo) & (pct < hi)).mean() * 100 for _, lo, hi in buckets]
        pos = x + (j - (len(models) - 1) / 2) * width
        ax.bar(pos, shares, width=width, color=MODEL_COLORS[model], edgecolor="white", linewidth=2,
               label=MODEL_LABELS[model])
        for xi, v in zip(pos, shares):
            if v >= 0.5:
                ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [b[0] for b in buckets], fontsize=8)
    ax.set_ylabel("% of top-10 recommendations", fontsize=8, color=MUTED)
    ax.set_ylim(0, 112)
    ax.set_title("Where recommendations come from, by movie popularity in the training window",
                 fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
