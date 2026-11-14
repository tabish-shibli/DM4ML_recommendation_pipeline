"""Stage 4 - Transformation.

1. Global time-based split. One cutoff timestamp for everybody: the model
   only ever sees the past and is tested on the future, as it would be in
   production. A second cutoff inside the training window gives a tuning split.
2. Derived datasets built from the training window only:
     user_activity_summary   one row per user
     item_popularity         one row per movie
     user_item_matrix        sparse users x movies rating matrix (+ index maps)
     item_genre_matrix       multi-hot genre flags per movie

Output: data/processed/splits/*.parquet, data/processed/derived/*
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import Config, load_config
from src.common.io import frame_fingerprint, write_json
from src.common.logging_setup import get_logger
from src.validation.schemas import CANONICAL_GENRES

log = get_logger(__name__)


def time_split(df: pd.DataFrame, holdout_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Everything up to the (1 - holdout_fraction) quantile of time goes to the first part."""
    cutoff_ts = int(np.quantile(df["timestamp"].to_numpy(), 1 - holdout_fraction))
    before = df[df["timestamp"] <= cutoff_ts].reset_index(drop=True)
    after = df[df["timestamp"] > cutoff_ts].reset_index(drop=True)
    return before, after, pd.Timestamp(cutoff_ts, unit="s", tz="UTC")


def split_stats(train: pd.DataFrame, test: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    tr_users, te_users = set(train["user_id"]), set(test["user_id"])
    tr_items = set(train["item_id"])
    warm = te_users & tr_users
    return {
        "cutoff": cutoff.isoformat(),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [train["event_time"].min().isoformat(), train["event_time"].max().isoformat()],
        "test_period": [test["event_time"].min().isoformat(), test["event_time"].max().isoformat()],
        "train_users": len(tr_users), "test_users": len(te_users),
        "warm_test_users": len(warm), "cold_start_test_users": len(te_users - tr_users),
        "test_rows_from_warm_users": int(test["user_id"].isin(warm).sum()),
        "test_rows_on_unseen_items": int((~test["item_id"].isin(tr_items)).sum()),
        "leakage_check_passed": bool(train["timestamp"].max() < test["timestamp"].min()),
    }


def user_activity_summary(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(n_ratings=("rating", "size"), avg_rating=("rating", "mean"), std_rating=("rating", "std"),
                first_event=("event_time", "min"), last_event=("event_time", "max"),
                n_positive=("rating", lambda r: int((r >= 4).sum()))).reset_index()
    out["std_rating"] = out["std_rating"].fillna(0.0)
    out["active_days"] = train.assign(d=train["event_time"].dt.floor("D")).groupby("user_id")["d"].nunique().values
    genre_lists = train[["user_id", "item_id"]].merge(items[["item_id", "genres"]], on="item_id")
    genre_lists = genre_lists.assign(genre=genre_lists["genres"].str.split("|")).explode("genre")
    out = out.merge(genre_lists.groupby("user_id")["genre"].nunique().rename("n_distinct_genres").reset_index(),
                    on="user_id", how="left")
    return out


def item_popularity(train: pd.DataFrame, items: pd.DataFrame, prior_weight: int) -> pd.DataFrame:
    mu = train["rating"].mean()
    g = train.groupby("item_id")
    pop = g.agg(n_ratings=("rating", "size"), n_users=("user_id", "nunique"), avg_rating=("rating", "mean"),
                rating_sum=("rating", "sum"), last_event=("event_time", "max")).reset_index()
    out = items[["item_id", "title"]].merge(pop, on="item_id", how="left")
    out["n_ratings"] = out["n_ratings"].fillna(0).astype(int)
    out["n_users"] = out["n_users"].fillna(0).astype(int)
    out["rating_sum"] = out["rating_sum"].fillna(0)
    out["bayesian_avg"] = (out["rating_sum"] + prior_weight * mu) / (out["n_ratings"] + prior_weight)
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    return out.drop(columns="rating_sum").sort_values("popularity_rank").reset_index(drop=True)


def build_matrix(train: pd.DataFrame, user_ids: np.ndarray | None = None, item_ids: np.ndarray | None = None):
    """CSR rating matrix plus the id <-> index maps that go with it."""
    user_ids = np.sort(train["user_id"].unique()) if user_ids is None else user_ids
    item_ids = np.sort(train["item_id"].unique()) if item_ids is None else item_ids
    u_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_index = pd.Series(np.arange(len(item_ids)), index=item_ids)
    rows = u_index.loc[train["user_id"]].to_numpy()
    cols = i_index.loc[train["item_id"]].to_numpy()
    m = sp.csr_matrix((train["rating"].to_numpy(np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return m, user_ids, item_ids


def item_genre_matrix(items: pd.DataFrame) -> pd.DataFrame:
    out = items[["item_id"]].copy()
    for g in CANONICAL_GENRES:
        out[f"genre_{g}"] = items["genres"].str.split("|").map(lambda gs, g=g: int(g in gs)).astype("int8")
    return out


def _plot_split(df: pd.DataFrame, cutoff: pd.Timestamp, inner: pd.Timestamp, path) -> None:
    monthly = df.set_index("event_time").resample("MS")["rating"].size()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(monthly.index, monthly.values, width=25, color="#4C72B0")
    ax.axvline(cutoff, color="#C44E52", linestyle="--", label=f"test cutoff {cutoff:%Y-%m-%d}")
    ax.axvline(inner, color="#DD8452", linestyle=":", label=f"tuning cutoff {inner:%Y-%m-%d}")
    ax.set_yscale("log")
    ax.set_ylabel("ratings per month (log)")
    ax.set_title("MovieLens 1M activity over time and the split points")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    pdir = cfg.path("processed")
    split_dir, derived_dir = pdir / "splits", pdir / "derived"
    split_dir.mkdir(exist_ok=True)
    derived_dir.mkdir(exist_ok=True)

    inter = pd.read_parquet(pdir / "interactions_clean.parquet")
    items = pd.read_parquet(pdir / "items_clean.parquet")
    scfg = cfg["split"]

    train, test, cutoff = time_split(inter, scfg["test_fraction"])
    train_inner, val, inner_cutoff = time_split(train, scfg["validation_fraction"])
    for name, part in (("train", train), ("test", test), ("train_inner", train_inner), ("val", val)):
        part.to_parquet(split_dir / f"{name}.parquet", index=False)

    stats = {"test_split": split_stats(train, test, cutoff),
             "tuning_split": split_stats(train_inner, val, inner_cutoff),
             "train_fingerprint": frame_fingerprint(train[["user_id", "item_id", "rating", "timestamp"]])}
    for k in ("test_split", "tuning_split"):
        if not stats[k]["leakage_check_passed"]:
            raise RuntimeError(f"{k}: training data overlaps the holdout period")
    write_json(stats, split_dir / "split_info.json")
    log.info("Test cutoff %s: train=%d test=%d (warm users %d, cold %d)", cutoff.date(), len(train), len(test),
             stats["test_split"]["warm_test_users"], stats["test_split"]["cold_start_test_users"])

    uas = user_activity_summary(train, items)
    uas.to_parquet(derived_dir / "user_activity_summary.parquet", index=False)
    pop = item_popularity(train, items, cfg["features"]["bayesian_prior_weight"])
    pop.to_parquet(derived_dir / "item_popularity.parquet", index=False)
    item_genre_matrix(items).to_parquet(derived_dir / "item_genre_matrix.parquet", index=False)

    # Matrix columns cover the whole catalogue so unrated movies still have an index
    m, uids, iids = build_matrix(train, item_ids=np.sort(items["item_id"].to_numpy()))
    sp.save_npz(derived_dir / "user_item_matrix.npz", m)
    pd.DataFrame({"user_id": uids, "row": np.arange(len(uids))}).to_csv(derived_dir / "user_index.csv", index=False)
    pd.DataFrame({"item_id": iids, "col": np.arange(len(iids))}).to_csv(derived_dir / "item_index.csv", index=False)
    density = m.nnz / (m.shape[0] * m.shape[1])
    log.info("User-item matrix %d x %d, %d ratings, density %.4f", m.shape[0], m.shape[1], m.nnz, density)

    fig_dir = cfg.path("reports", "figures")
    _plot_split(inter, cutoff, inner_cutoff, fig_dir / "split_timeline.png")

    summary = {"stage": "transformation", **stats,
               "matrix": {"users": m.shape[0], "items": m.shape[1], "nnz": int(m.nnz), "density": density},
               "derived_rows": {"user_activity_summary": len(uas), "item_popularity": len(pop)}}
    write_json(summary, cfg.path("reports") / "transformation" / "transformation_summary.json")
    return summary


if __name__ == "__main__":
    run()
