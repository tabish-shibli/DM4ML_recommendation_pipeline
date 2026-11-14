"""Prediction path: top-K recommendations for one user.

    python -m src.serving.recommend --user-id 953 --model hybrid --k 10

Reads user and item features from the online store (SQLite), the same values
that were written to the offline snapshot used in training. Before scoring it
checks that the online store holds the snapshot the model was trained on, so
training/serving skew is caught instead of silently producing odd results.
Users with no history fall back to the popularity model.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import load_config
from src.common.io import read_json
from src.features.store import FeatureStore
from src.training.models import HybridRecommender, Recommender
from src.validation.schemas import CANONICAL_GENRES


class SnapshotMismatch(RuntimeError):
    pass


def recommend(user_id: int, model_name: str = "hybrid", k: int = 10, cfg=None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config()
    store = FeatureStore(cfg)
    models_dir = cfg.path("models")
    meta = read_json(models_dir / model_name / "metadata.json")
    online = store.online_info()
    if online["reference_time"] != meta["feature_reference_time"]:
        raise SnapshotMismatch(f"Online store is at {online['reference_time']} but {model_name} was trained on "
                               f"features as of {meta['feature_reference_time']}. Re-run the feature stage.")

    model: Recommender = Recommender.load(models_dir / model_name / "model.pkl")
    derived = cfg.path("processed") / "derived"
    seen = sp.load_npz(derived / "user_item_matrix.npz").tocsr()
    profile = store.online_user_features([user_id])
    info = {"user_id": user_id, "model": model_name, "feature_snapshot_as_of": online["reference_time"]}

    row = int(model.user_rows([user_id])[0])
    if row < 0:
        model = Recommender.load(models_dir / "popularity" / "model.pkl")
        info["note"] = "no training history for this user, fell back to popularity"
        scores = model.score(np.array([0]))[0]
        seen_mask = np.zeros(len(model.item_ids), dtype=bool)
    else:
        if isinstance(model, HybridRecommender):
            aff = store.online_genre_affinity([user_id])
            vec = aff[[f"affinity_{g}" for g in CANONICAL_GENRES]].to_numpy(dtype=float)
            scores = model.score(np.array([row]), affinity=vec)[0]
        else:
            scores = model.score(np.array([row]))[0]
        seen_mask = seen[row].toarray()[0] > 0
        if not profile.empty:
            info["profile"] = profile.iloc[0].to_dict()

    item_feats = store.online_item_features(model.item_ids).set_index("item_id").reindex(model.item_ids)
    scores = np.where(seen_mask | (item_feats["item_popularity"].to_numpy() == 0), -np.inf, scores)
    top = np.argsort(-scores, kind="stable")[:k]

    items = pd.read_parquet(cfg.path("processed") / "items_clean.parquet").set_index("item_id")
    ids = model.item_ids[top]
    out = pd.DataFrame({"rank": np.arange(1, k + 1), "item_id": ids, "title": items.loc[ids, "title"].to_numpy(),
                        "year": items.loc[ids, "release_year"].to_numpy(), "genres": items.loc[ids, "genres"].to_numpy(),
                        "score": scores[top].round(3),
                        "ratings_in_training": item_feats.loc[ids, "item_popularity"].to_numpy()})
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-K movie recommendations for one user")
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--model", default="hybrid", choices=["popularity", "matrix_factorization", "hybrid"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    recs, info = recommend(args.user_id, args.model, args.k)
    print(f"\nUser {info['user_id']}  |  model: {info['model']}  |  features as of {info['feature_snapshot_as_of']}")
    if "note" in info:
        print(f"Note: {info['note']}")
    if "profile" in info:
        p = info["profile"]
        print(f"Profile: {p['user_gender']}, {p['user_age_group']}, {p['user_occupation']}; "
              f"{p['user_activity_count']} ratings, mean {p['user_avg_rating']:.2f}, likes {p['user_preferred_genre']}")
    print()
    with pd.option_context("display.width", 160, "display.max_colwidth", 45):
        print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
