"""Draws reports/figures/architecture.png.  Run: python scripts/draw_architecture.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
STAGE_FILL, STORE_FILL = "#eaf2fc", "#f4f3f0"
ACCENT = "#2a78d6"

fig, ax = plt.subplots(figsize=(17.6, 7.6))
ax.set_xlim(0, 176)
ax.set_ylim(0, 76)
ax.axis("off")


def box(cx, cy, w, h, title, lines, fill, edge=EDGE, lw=1.1, title_size=10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    top = cy + h / 2 - 2.6
    ax.text(cx, top, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(cx, top - 3.1 - i * 2.4, line, ha="center", va="center", fontsize=7.2, color=MUTED)


def arrow(x1, y1, x2, y2, color=EDGE, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color,
                                 linewidth=1.1, linestyle=ls, shrinkA=0, shrinkB=0))


xs = [27 + 22.5 * i for i in range(7)]
W, SY, SH, DY, DH = 20.5, 44, 15.5, 19, 10.5

stages = [
    ("1  Ingestion", ["download + md5 check", "parse .dat -> landing CSV", "seeded dirty-feed simulation", "manifest + defect log"]),
    ("2  Validation", ["Pandera contracts", "types, ranges, timestamps,", "duplicates, FK integrity", "quality gate: <=5% rejected"]),
    ("3  Preparation", ["gender/age/occupation", "standardised; titles fixed", "genres normalised", "re-rates -> latest kept"]),
    ("4  Transformation", ["global time split", "(+ inner tuning split)", "user activity, item", "popularity, U x I matrix"]),
    ("5  Features", ["registry (YAML) contract", "point-in-time snapshots", "user / item / interaction", "+ on-demand genre match"]),
    ("6  Training", ["Popularity baseline", "ALS matrix factorisation", "Hybrid (MF+pop+genre)", "tuned on inner split"]),
    ("7  Evaluation", ["P/R/NDCG/Hit @5,@10", "coverage, diversity,", "RMSE/MAE, bootstrap CI", "cold-start fallback"]),
]
stores = [
    ("data/raw/landing", ["users, items, interactions", "_manifest.json"]),
    ("validated + errors", ["clean rows (parquet)", "*_rejected.csv + reasons"]),
    ("data/processed", ["*_clean.parquet", "preparation summary"]),
    ("splits + derived", ["train / test / val", "matrix .npz + index maps"]),
    ("feature store", ["offline: parquet snapshots", "online: SQLite"]),
    ("models/ + MLflow", ["model.pkl + metadata", "params, metrics, artifacts"]),
    ("reports/", ["validation, catalog,", "comparison, figures"]),
]

# orchestration band
ax.add_patch(FancyBboxPatch((xs[0] - W / 2, 58), xs[-1] - xs[0] + W, 8, boxstyle="round,pad=0.25,rounding_size=1.2",
                            facecolor="white", edgecolor=ACCENT, linewidth=1.6, linestyle=(0, (5, 3))))
ax.text((xs[0] + xs[-1]) / 2, 63.6, "Prefect flow  'movielens-recsys-pipeline'  (dags/recommendation_flow.py)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT)
ax.text((xs[0] + xs[-1]) / 2, 60.4, "tasks run strictly in dependency order  |  retries on ingest  |  "
        "a failed quality gate stops everything downstream  |  weekly cron via --serve  |  run_pipeline.py = same stages without a server",
        ha="center", va="center", fontsize=7.8, color=MUTED)
for x in xs:
    arrow(x, 57.7, x, SY + SH / 2 + 0.4, color=ACCENT, ls=(0, (2, 2)))

for (title, lines), x in zip(stages, xs):
    box(x, SY, W, SH, title, lines, STAGE_FILL, edge=ACCENT if title.startswith("2") else EDGE,
        lw=1.8 if title.startswith("2") else 1.1)
for (title, lines), x in zip(stores, xs):
    box(x, DY, W, DH, title, lines, STORE_FILL, title_size=9.2)

for i, x in enumerate(xs):
    arrow(x, SY - SH / 2 - 0.3, x, DY + DH / 2 + 0.4)                       # stage writes its zone
    if i + 1 < len(xs):
        arrow(x + W / 2 - 1.5, DY + DH / 2 + 0.4, xs[i + 1] - W / 2 + 1.5, SY - SH / 2 - 0.3)  # next stage reads it

# source
box(7.5, SY, 11.5, 12, "Source", ["MovieLens 1M", "GroupLens", "md5 pinned"], "white", title_size=9.5)
arrow(13.6, SY, xs[0] - W / 2 - 0.4, SY)

# evaluation also writes back to MLflow
arrow(xs[6] - W / 2 + 2, DY - DH / 2 - 0.4, xs[5] + W / 2 - 2, DY - DH / 2 - 0.4, color=MUTED, ls=(0, (3, 2)))
ax.text(xs[6], DY - DH / 2 - 2.6, "test metrics logged back to MLflow", ha="center", fontsize=7, color=MUTED)

# serving
sx, sy = (xs[4] + xs[5]) / 2, 4.2
ax.add_patch(FancyBboxPatch((sx - 27, sy - 3.2), 54, 6.4, boxstyle="round,pad=0.25,rounding_size=1.2",
                            facecolor="white", edgecolor=EDGE, linewidth=1.1))
ax.text(sx, sy + 0.9, "Serving: python -m src.serving.recommend --user-id N", ha="center", fontsize=8.6,
        fontweight="bold", color=INK)
ax.text(sx, sy - 1.6, "online features + saved model, snapshot check, popularity fallback for new users",
        ha="center", fontsize=7.2, color=MUTED)
arrow(xs[4], DY - DH / 2 - 0.4, xs[4] + 3, sy + 3.5)
arrow(xs[5], DY - DH / 2 - 0.4, xs[5] - 3, sy + 3.5)

ax.text(1, 73.5, "End-to-end data management pipeline for a movie recommender", fontsize=13, fontweight="bold", color=INK)
ax.text(1, 70.8, "Each stage reads only what the previous stage wrote, so any stage can be re-run on its own "
        "and the zones double as an audit trail.", fontsize=8.6, color=MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
