"""Stage 6 - Model training (with tuning) and experiment tracking.

1. Tuning, on the inner split only (train_inner -> val, features from the
   `tuning` snapshot). The MF regularisation is picked by validation RMSE,
   because rating error is what MF optimises. The hybrid weights are then
   picked by validation NDCG@10.
2. Final fit of all three models on the full training window with the
   `training` feature snapshot and the chosen settings.

Every configuration tried is an MLflow run nested under one parent run.
Models are saved to models/<name>/ together with their metadata.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import FEATURE_REGISTRY, PROJECT_ROOT, Config, load_config
from src.common.io import read_json, utc_now, write_json
from src.common.logging_setup import get_logger
from src.evaluation.ranking import EvalSet, build_eval_set, rating_metrics, ranking_metrics, top_k
from src.features.store import FeatureStore
from src.training.models import ALSMatrixFactorization, HybridRecommender, PopularityRecommender
from src.training.tracking import code_version, setup_mlflow
from src.transformation.transform import build_matrix

log = get_logger(__name__)


@dataclass
class Context:
    """Everything a model needs for one train/holdout pair."""
    ratings: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    features: dict
    eval_set: EvalSet
    candidates: np.ndarray
    genre_vectors: np.ndarray
    item_user_share: np.ndarray


def make_context(window: pd.DataFrame, holdout: pd.DataFrame, item_ids: np.ndarray, features: dict,
                 threshold: int, ratings: sp.csr_matrix | None = None, user_ids: np.ndarray | None = None) -> Context:
    if ratings is None:
        ratings, user_ids, item_ids = build_matrix(window, item_ids=item_ids)
    pop = features["item_features"].set_index("item_id")["item_popularity"].reindex(item_ids).fillna(0).to_numpy()
    genres = features["item_genres"].set_index("item_id").reindex(item_ids).fillna(0).to_numpy(dtype=float)
    return Context(ratings=ratings, user_ids=user_ids, item_ids=item_ids, features=features,
                   eval_set=build_eval_set(holdout, user_ids, item_ids, threshold),
                   candidates=pop > 0, genre_vectors=genres, item_user_share=pop / len(user_ids))


def score_model(model, ctx: Context, k_values=(10,)) -> dict:
    topk = top_k(model, ctx.eval_set.user_rows, ctx.ratings, ctx.candidates, max(k_values))
    summary, _ = ranking_metrics(topk, ctx.eval_set, k_values, ctx.genre_vectors, ctx.item_user_share,
                                 int(ctx.candidates.sum()))
    summary.update(rating_metrics(model, ctx.eval_set))
    return summary


def tune(cfg: Config, ctx: Context) -> tuple[dict, pd.DataFrame]:
    tcfg, mfcfg = cfg["training"]["tuning"], cfg["training"]["matrix_factorization"]
    rows = []

    # reference point: the baseline on the same validation users
    pop = PopularityRecommender().fit(ctx.features["item_features"], ctx.user_ids, ctx.item_ids)
    m = score_model(pop, ctx)
    rows.append({"model": pop.name, **{f"val_{k}": v for k, v in m.items()}})
    log.info("tune baseline popularity  val RMSE=%.4f  val NDCG@10=%.4f", m["rmse"], m["ndcg_at_10"])

    best_mf, best_rmse = None, np.inf
    for reg in tcfg["reg_grid"]:
        mf = ALSMatrixFactorization(mfcfg["factors"], reg, mfcfg["iterations"], cfg.seed)
        mf.fit(ctx.ratings, ctx.user_ids, ctx.item_ids)
        m = score_model(mf, ctx)
        with mlflow.start_run(run_name=f"tune-mf-reg{reg}", nested=True):
            mlflow.set_tags({"phase": "tuning", "model": mf.name})
            mlflow.log_params(mf.params())
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()} | {"fit_seconds": mf.fit_seconds})
        rows.append({"model": mf.name, **mf.params(), **{f"val_{k}": v for k, v in m.items()}})
        log.info("tune MF reg=%.3f  val RMSE=%.4f  val NDCG@10=%.4f", reg, m["rmse"], m["ndcg_at_10"])
        if m["rmse"] < best_rmse:
            best_mf, best_rmse = mf, m["rmse"]

    grid = tcfg["hybrid_grid"]
    base = HybridRecommender(best_mf).fit(ctx.features["item_features"], ctx.features["user_genre_affinity"],
                                          ctx.features["item_genres"])
    metric = tcfg["selection_metric"]
    best_w, best_val = None, -np.inf
    for w_mf, w_pop, w_genre in itertools.product(grid["mf_weight"], grid["popularity_weight"], grid["genre_weight"]):
        hy = base.with_weights(w_mf, w_pop, w_genre)
        m = score_model(hy, ctx)
        with mlflow.start_run(run_name=f"tune-hybrid-pop{w_pop}-genre{w_genre}", nested=True):
            mlflow.set_tags({"phase": "tuning", "model": hy.name})
            mlflow.log_params(hy.params())
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
        rows.append({"model": hy.name, **hy.params(), **{f"val_{k}": v for k, v in m.items()}})
        log.info("tune hybrid pop=%.2f genre=%.2f  val NDCG@10=%.4f", w_pop, w_genre, m["ndcg_at_10"])
        if m[metric] > best_val:
            best_w, best_val = (w_mf, w_pop, w_genre), m[metric]

    best = {"mf_reg": best_mf.reg, "mf_val_rmse": best_rmse, "hybrid_mf_weight": best_w[0],
            "hybrid_popularity_weight": best_w[1], "hybrid_genre_weight": best_w[2],
            f"hybrid_val_{metric}": best_val}
    return best, pd.DataFrame(rows)


def _save(model, cfg: Config, meta: dict) -> dict:
    out = cfg.path("models") / model.name
    model.save(out / "model.pkl")
    meta = {"name": model.name, "class": type(model).__name__, "params": model.params(),
            "fit_seconds": model.fit_seconds, "trained_at": utc_now(), **meta}
    write_json(meta, out / "metadata.json")
    return meta


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    pdir = cfg.path("processed")
    derived = pdir / "derived"
    split_info = read_json(pdir / "splits" / "split_info.json")
    store = FeatureStore(cfg)
    threshold = cfg["evaluation"]["relevance_threshold"]
    item_ids = pd.read_csv(derived / "item_index.csv")["item_id"].to_numpy()

    setup_mlflow(cfg)
    reports = cfg.path("reports", "training")

    with mlflow.start_run(run_name=f"training-{pd.Timestamp.now():%Y%m%d-%H%M}") as parent:
        mlflow.set_tags({"phase": "training", "dataset": cfg["ingestion"]["source_name"],
                         "code_version": code_version(), "train_fingerprint": split_info["train_fingerprint"]})
        mlflow.log_params({"test_cutoff": split_info["test_split"]["cutoff"],
                           "tuning_cutoff": split_info["tuning_split"]["cutoff"],
                           "relevance_threshold": threshold, "seed": cfg.seed})
        mlflow.log_artifact(str(PROJECT_ROOT / "configs" / "pipeline.yaml"), "config")
        mlflow.log_artifact(str(FEATURE_REGISTRY), "config")
        mlflow.log_artifact(str(cfg.path("reports") / "validation" / "validation_report.md"), "data_quality")

        # ---------------- tuning on the inner split
        tune_ctx = make_context(pd.read_parquet(pdir / "splits" / "train_inner.parquet"),
                                pd.read_parquet(pdir / "splits" / "val.parquet"),
                                item_ids, store.load("tuning"), threshold)
        log.info("Tuning on %d ratings, validating on %d users", tune_ctx.ratings.nnz, len(tune_ctx.eval_set.user_rows))
        best, tuning_table = tune(cfg, tune_ctx)
        tuning_table.to_csv(reports / "tuning_results.csv", index=False)
        write_json(best, reports / "best_params.json")
        mlflow.log_artifact(str(reports / "tuning_results.csv"), "tuning")
        mlflow.log_params({f"best_{k}": v for k, v in best.items() if not k.startswith(("mf_val", "hybrid_val"))})
        log.info("Selected: %s", best)

        # ---------------- final fit on the full training window
        feats = store.load("training")
        ratings = sp.load_npz(derived / "user_item_matrix.npz").tocsr()
        user_ids = pd.read_csv(derived / "user_index.csv")["user_id"].to_numpy()
        mfcfg = cfg["training"]["matrix_factorization"]
        lineage = {"feature_snapshot": "training", "feature_reference_time": store.metadata("training")["reference_time"],
                   "train_fingerprint": split_info["train_fingerprint"], "train_rows": int(ratings.nnz)}

        pop = PopularityRecommender().fit(feats["item_features"], user_ids, item_ids)
        mf = ALSMatrixFactorization(mfcfg["factors"], best["mf_reg"], mfcfg["iterations"], cfg.seed)
        mf.fit(ratings, user_ids, item_ids)
        hy = HybridRecommender(mf, best["hybrid_mf_weight"], best["hybrid_popularity_weight"],
                               best["hybrid_genre_weight"]).fit(feats["item_features"], feats["user_genre_affinity"],
                                                                feats["item_genres"])
        registry = {}
        for model in (pop, mf, hy):
            with mlflow.start_run(run_name=model.name, nested=True) as run_:
                mlflow.set_tags({"phase": "final", "model": model.name})
                mlflow.log_params(model.params() or {"strategy": "most_rated"})
                mlflow.log_metric("fit_seconds", model.fit_seconds)
                for h in getattr(model, "history", []):
                    mlflow.log_metric("train_rmse", h["train_rmse"], step=h["iteration"])
                meta = _save(model, cfg, {"mlflow_run_id": run_.info.run_id, "parent_run_id": parent.info.run_id,
                                          **lineage})
                mlflow.log_artifact(str(cfg.path("models") / model.name / "model.pkl"), "model")
                mlflow.log_artifact(str(cfg.path("models") / model.name / "metadata.json"), "model")
                registry[model.name] = meta
                log.info("Trained %-22s in %.1fs (run %s)", model.name, model.fit_seconds, run_.info.run_id[:8])

        write_json({"parent_run_id": parent.info.run_id, "models": registry}, cfg.path("models") / "registry.json")
    return {"stage": "training", "best_params": best, "models": list(registry)}


if __name__ == "__main__":
    run()
