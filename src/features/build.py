"""Feature computation. Pure functions: interactions in, feature tables out.

Everything is computed as of a reference time from interactions on or before
that time, so the same code produces leak-free features for tuning (as of the
tuning cutoff) and for training/serving (as of the test cutoff).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.schemas import CANONICAL_GENRES

DAY = pd.Timedelta(days=1)


def _genre_long(window: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """One row per (rating, genre) - a movie with 3 genres contributes 3 rows."""
    df = window[["user_id", "item_id", "rating"]].merge(items[["item_id", "genres"]], on="item_id")
    df = df.assign(genre=df["genres"].str.split("|")).explode("genre")
    return df[df["genre"].isin(CANONICAL_GENRES)]


def item_features(window: pd.DataFrame, items: pd.DataFrame, ref: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    mu = window["rating"].mean()
    m = cfg["bayesian_prior_weight"]
    g = window.groupby("item_id")
    agg = g.agg(item_popularity=("rating", "size"), item_avg_rating=("rating", "mean"),
                item_rating_std=("rating", "std"), rating_sum=("rating", "sum"),
                last_event=("event_time", "max")).reset_index()
    recent = window[window["event_time"] > ref - cfg["trending_window_days"] * DAY]
    trending = recent.groupby("item_id").size().rename("item_trending_count").reset_index()

    out = items[["item_id", "release_year", "n_genres"]].merge(agg, on="item_id", how="left")
    out = out.merge(trending, on="item_id", how="left")
    out["item_popularity"] = out["item_popularity"].fillna(0).astype("int64")
    out["item_trending_count"] = out["item_trending_count"].fillna(0).astype("int64")
    out["item_popularity_pct"] = out["item_popularity"].rank(pct=True, method="average").astype("float64")
    out["item_bayesian_rating"] = (out["rating_sum"].fillna(0) + m * mu) / (out["item_popularity"] + m)
    out["item_rating_std"] = out["item_rating_std"].fillna(0.0)
    out["item_days_since_last"] = (ref - out["last_event"]) / DAY
    out["item_release_year"] = out["release_year"].astype("int64")
    out["item_age_years"] = (ref.year - out["item_release_year"]).astype("int64")
    out["item_n_genres"] = out["n_genres"].astype("int64")
    cols = ["item_id", "item_popularity", "item_popularity_pct", "item_avg_rating", "item_bayesian_rating",
            "item_rating_std", "item_days_since_last", "item_trending_count", "item_release_year",
            "item_age_years", "item_n_genres"]
    return out[cols].sort_values("item_id").reset_index(drop=True)


def item_genres(items: pd.DataFrame) -> pd.DataFrame:
    out = items[["item_id"]].copy()
    lists = items["genres"].str.split("|")
    for g in CANONICAL_GENRES:
        out[f"genre_{g}"] = lists.map(lambda gs, g=g: int(g in gs)).astype("int64")
    return out.sort_values("item_id").reset_index(drop=True)


def user_genre_affinity(window: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    long = _genre_long(window, items)
    counts = long.pivot_table(index="user_id", columns="genre", values="item_id", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=CANONICAL_GENRES, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0).astype("float64")
    share.columns = [f"affinity_{g}" for g in CANONICAL_GENRES]
    return share.reset_index()


def user_features(window: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, item_feats: pd.DataFrame,
                  ref: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    g = window.groupby("user_id")
    out = g.agg(user_activity_count=("rating", "size"), user_avg_rating=("rating", "mean"),
                user_rating_std=("rating", "std"),
                user_positive_share=("rating", lambda r: float((r >= cfg["positive_rating_threshold"]).mean())),
                first_event=("event_time", "min"), last_event=("event_time", "max")).reset_index()
    out["user_rating_std"] = out["user_rating_std"].fillna(0.0)
    out["user_days_since_last"] = (ref - out["last_event"]) / DAY
    out["user_tenure_days"] = (out["last_event"] - out["first_event"]) / DAY

    long = _genre_long(window, items)
    gstats = long.groupby(["user_id", "genre"]).agg(n=("rating", "size"), avg=("rating", "mean")).reset_index()
    pref = gstats.sort_values(["user_id", "n", "avg"], ascending=[True, False, False]).drop_duplicates("user_id")
    out = out.merge(pref[["user_id", "genre"]].rename(columns={"genre": "user_preferred_genre"}), on="user_id", how="left")

    p = gstats.assign(p=gstats["n"] / gstats.groupby("user_id")["n"].transform("sum"))
    ent = (-(p["p"] * np.log(p["p"]))).groupby(p["user_id"]).sum() / np.log(len(CANONICAL_GENRES))
    out = out.merge(ent.rename("user_genre_entropy").reset_index(), on="user_id", how="left")

    mainstream = window[["user_id", "item_id"]].merge(item_feats[["item_id", "item_popularity_pct"]], on="item_id")
    out = out.merge(mainstream.groupby("user_id")["item_popularity_pct"].mean().rename("user_mainstreamness").reset_index(),
                    on="user_id", how="left")

    demo = users[["user_id", "gender", "age_group", "occupation"]].rename(
        columns={"gender": "user_gender", "age_group": "user_age_group", "occupation": "user_occupation"})
    out = out.merge(demo, on="user_id", how="left")
    for c in ("user_gender", "user_age_group", "user_occupation", "user_preferred_genre"):
        out[c] = out[c].fillna("unknown").astype(str)
    out["user_activity_count"] = out["user_activity_count"].astype("int64")
    cols = ["user_id", "user_gender", "user_age_group", "user_occupation", "user_activity_count",
            "user_avg_rating", "user_rating_std", "user_positive_share", "user_days_since_last",
            "user_tenure_days", "user_preferred_genre", "user_genre_entropy", "user_mainstreamness"]
    return out[cols].sort_values("user_id").reset_index(drop=True)


def interaction_features(window: pd.DataFrame, user_feats: pd.DataFrame, ref: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    out = window[["user_id", "item_id", "rating", "event_time"]].merge(
        user_feats[["user_id", "user_avg_rating"]], on="user_id", how="left")
    out["ix_days_before_reference"] = (ref - out["event_time"]) / DAY
    out["ix_recency_weight"] = 0.5 ** (out["ix_days_before_reference"] / cfg["recency_half_life_days"])
    out["ix_rating_centered"] = out["rating"] - out["user_avg_rating"]
    out["ix_is_positive"] = (out["rating"] >= cfg["positive_rating_threshold"]).astype("int64")
    return out.drop(columns="user_avg_rating")


def genre_match(affinity: np.ndarray, genre_flags: np.ndarray) -> np.ndarray:
    """On-demand feature ui_genre_match for every (user, item) combination.

    affinity: users x 18 shares, genre_flags: items x 18 multi-hot.
    Returns users x items. A movie scores high when its genres are ones the
    user watches a lot; dividing by the genre count stops movies tagged with
    many genres from winning by default.
    """
    n = np.maximum(genre_flags.sum(axis=1), 1)
    return (affinity @ genre_flags.T) / n[None, :]


def build_all(window: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, ref: pd.Timestamp,
              cfg: dict) -> dict[str, pd.DataFrame]:
    window = window[window["event_time"] <= ref]
    itf = item_features(window, items, ref, cfg)
    uf = user_features(window, users, items, itf, ref, cfg)
    return {
        "user_features": uf,
        "user_genre_affinity": user_genre_affinity(window, items),
        "item_features": itf,
        "item_genres": item_genres(items),
        "interaction_features": interaction_features(window, uf, ref, cfg),
    }
