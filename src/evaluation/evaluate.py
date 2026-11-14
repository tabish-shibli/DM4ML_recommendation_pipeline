"""Stage 7 - Model evaluation on the test period.

For every trained model:
  - Precision, Recall, NDCG, Hit rate at K = 5 and 10 (warm users)
  - Catalogue coverage, intra-list diversity (genres), novelty
  - RMSE / MAE for models that predict star ratings
  - Precision/Recall curves for K = 1..20
  - Paired bootstrap confidence intervals against the baseline
Cold-start users (no training history) are reported separately: every model
falls back to popularity for them.

Results go to reports/evaluation/ and reports/figures/, and the test metrics
are written back into each model's MLflow run.
"""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import Config, load_config
from src.common.io import df_to_markdown, read_json, utc_now, write_json
from src.common.logging_setup import get_logger
from src.evaluation import metrics as M
from src.evaluation import plots
from src.evaluation.ranking import build_eval_set, rating_metrics, ranking_metrics, top_k
from src.features.store import FeatureStore
from src.training.models import Recommender
from src.training.tracking import setup_mlflow

log = get_logger(__name__)
CURVE_K = 20


def _cold_start(test: pd.DataFrame, train_users: np.ndarray, pop_model, item_ids, candidates, threshold, k):
    """Users with no training history get the popularity list; score that list on them."""
    cold = test[~test["user_id"].isin(train_users)]
    pos = cold[cold["rating"] >= threshold]
    if pos.empty:
        return {}
    users = np.sort(pos["user_id"].unique())
    i_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    u_pos = pd.Series(np.arange(len(users)), index=users)
    relevant = np.zeros((len(users), len(item_ids)), dtype=bool)
    relevant[u_pos.loc[pos["user_id"]].to_numpy(), i_pos.loc[pos["item_id"]].to_numpy()] = True
    s = np.where(candidates, pop_model._score, -np.inf)
    top = np.argsort(-s, kind="stable")[:k]
    topk = np.tile(top, (len(users), 1))
    hits = M.hits_matrix(topk, relevant)
    n_rel = relevant.sum(axis=1)
    return {"cold_users_evaluated": len(users),
            f"precision_at_{k}": float(M.precision_at_k(hits, k).mean()),
            f"recall_at_{k}": float(M.recall_at_k(hits, n_rel, k).mean()),
            f"ndcg_at_{k}": float(M.ndcg_at_k(hits, n_rel, k).mean())}


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    ecfg = cfg["evaluation"]
    pdir, derived = cfg.path("processed"), cfg.path("processed") / "derived"
    out_dir = cfg.path("reports", "evaluation")
    fig_dir = cfg.path("reports", "figures")

    registry = read_json(cfg.path("models") / "registry.json")
    models: dict[str, Recommender] = {name: Recommender.load(cfg.path("models") / name / "model.pkl")
                                      for name in registry["models"]}
    test = pd.read_parquet(pdir / "splits" / "test.parquet")
    ratings = sp.load_npz(derived / "user_item_matrix.npz").tocsr()
    user_ids = pd.read_csv(derived / "user_index.csv")["user_id"].to_numpy()
    item_ids = pd.read_csv(derived / "item_index.csv")["item_id"].to_numpy()
    feats = FeatureStore(cfg).load("training")
    items = pd.read_parquet(pdir / "items_clean.parquet").set_index("item_id")

    ifeat = feats["item_features"].set_index("item_id").reindex(item_ids)
    candidates = (ifeat["item_popularity"] > 0).to_numpy()
    genre_vectors = feats["item_genres"].set_index("item_id").reindex(item_ids).fillna(0).to_numpy(dtype=float)
    user_share = ifeat["item_popularity"].fillna(0).to_numpy() / len(user_ids)
    pop_pct = ifeat["item_popularity_pct"].fillna(0).to_numpy()
    es = build_eval_set(test, user_ids, item_ids, ecfg["relevance_threshold"])
    log.info("Evaluating %d warm test users (%d relevant items, %d rated pairs)", len(es.user_rows),
             int(es.n_relevant.sum()), len(es.rating_true))

    rows, per_user, curves, rec_pop, topk_store = [], {}, [], {}, {}
    for name, model in models.items():
        topk = top_k(model, es.user_rows, ratings, candidates, CURVE_K)
        topk_store[name] = topk
        summary, pu = ranking_metrics(topk, es, ecfg["k_values"], genre_vectors, user_share, int(candidates.sum()))
        summary.update(rating_metrics(model, es))
        k = ecfg["primary_k"]
        summary[f"head_share_at_{k}"] = float((pop_pct[topk[:, :k]] >= 0.9).mean())
        rows.append({"model": name, **summary})
        per_user[name] = pu
        rec_pop[name] = pop_pct[topk[:, :k]].ravel()
        hits = M.hits_matrix(topk, es.relevant)
        for kk in range(1, CURVE_K + 1):
            curves.append({"model": name, "k": kk, "precision": M.precision_at_k(hits, kk).mean(),
                           "recall": M.recall_at_k(hits, es.n_relevant, kk).mean()})
        log.info("%-22s P@10=%.4f R@10=%.4f NDCG@10=%.4f cov=%.3f div=%.3f %s", name, summary["precision_at_10"],
                 summary["recall_at_10"], summary["ndcg_at_10"], summary["coverage_at_10"],
                 summary["diversity_at_10"], f"RMSE={summary['rmse']:.4f}" if "rmse" in summary else "")

    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(out_dir / "model_comparison.csv")
    pd.concat([df.assign(model=n) for n, df in per_user.items()]).to_parquet(out_dir / "per_user_metrics.parquet")
    curves = pd.DataFrame(curves)
    curves.to_csv(out_dir / "precision_recall_curves.csv", index=False)

    # paired bootstrap on per-user NDCG@10 against the baseline
    primary = f"ndcg_at_{ecfg['primary_k']}"
    base = per_user["popularity"][primary].to_numpy()
    significance = []
    for name in models:
        if name == "popularity":
            continue
        mean, lo, hi = M.bootstrap_diff_ci(per_user[name][primary].to_numpy(), base, ecfg["bootstrap_samples"], cfg.seed)
        significance.append({"comparison": f"{name} - popularity", "metric": primary, "mean_difference": mean,
                             "ci95_low": lo, "ci95_high": hi, "significant": "yes" if lo > 0 or hi < 0 else "no"})
    significance = pd.DataFrame(significance)
    significance.to_csv(out_dir / "significance.csv", index=False)

    cold = _cold_start(test, user_ids, models["popularity"], item_ids, candidates, ecfg["relevance_threshold"],
                       ecfg["primary_k"])

    # a few real users, to see what the numbers mean
    examples = []
    rng = np.random.default_rng(cfg.seed)
    for idx in rng.choice(len(es.user_rows), 3, replace=False):
        uid = int(es.user_ids[idx])
        hist = test[test["user_id"] == uid]
        for name, topk in topk_store.items():
            recs = [int(item_ids[c]) for c in topk[idx, :5]]
            liked = set(hist.loc[hist["rating"] >= ecfg["relevance_threshold"], "item_id"])
            examples.append({"user_id": uid, "model": name,
                             "top_5": "; ".join(f"{items.loc[i, 'title']}{' *' if i in liked else ''}" for i in recs)})
    examples = pd.DataFrame(examples)

    plots.metric_bars(table, ecfg["primary_k"], fig_dir / "metrics_comparison.png")
    plots.precision_recall_curves(curves, fig_dir / "precision_recall_at_k.png")
    plots.popularity_profile(rec_pop, fig_dir / "popularity_profile.png")
    _write_markdown(table, significance, cold, examples, es, ecfg, out_dir)

    setup_mlflow(cfg)
    for name, meta in registry["models"].items():
        with mlflow.start_run(run_id=meta["mlflow_run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in table.loc[name].dropna().items()})
    with mlflow.start_run(run_id=registry["parent_run_id"]):
        for f in ("model_comparison.md", "model_comparison.csv", "significance.csv"):
            mlflow.log_artifact(str(out_dir / f), "evaluation")
        for f in ("metrics_comparison.png", "precision_recall_at_k.png", "popularity_profile.png"):
            mlflow.log_artifact(str(fig_dir / f), "figures")

    result = {"stage": "evaluation", "evaluated_at": utc_now(), "warm_users": len(es.user_rows),
              "metrics": table.reset_index().to_dict("records"), "cold_start": cold,
              "significance": significance.to_dict("records")}
    write_json(result, out_dir / "evaluation_summary.json")
    return result


def _write_markdown(table, significance, cold, examples, es, ecfg, out_dir) -> None:
    k = ecfg["primary_k"]
    order = ["precision_at_{k}", "recall_at_{k}", "ndcg_at_{k}", "hit_rate_at_{k}", "coverage_at_{k}",
             "diversity_at_{k}", "personalization_at_{k}", "novelty_at_{k}"]
    main = table[[c.format(k=k) for c in order] + [f"head_share_at_{k}"]].reset_index()
    k5 = table[[c.format(k=5) for c in order[:4]]].reset_index()
    rating = table[["rmse", "mae"]].dropna().reset_index()
    lines = [
        "# Model Evaluation Results", "",
        f"Test period: ratings after the time cutoff. Evaluated users: {len(es.user_rows):,} warm users "
        f"(seen in training, at least one test rating >= {ecfg['relevance_threshold']}). "
        f"Relevant item = test rating >= {ecfg['relevance_threshold']}. Movies already rated in training are "
        "never recommended.", "",
        f"## Ranking metrics at K = {k}", "", df_to_markdown(main), "",
        f"`coverage` is the share of recommendable movies that appear in anyone's list. `diversity` is the mean "
        "pairwise genre dissimilarity inside a list. `personalization` is 1 minus the average overlap between two "
        "users' lists. `novelty` is the mean self-information (-log2 of the share of users who rated the movie). "
        f"`head_share_at_{k}` is the share of recommendations drawn from the 10% most-rated movies.", "",
        "## Ranking metrics at K = 5", "", df_to_markdown(k5), "",
        "## Rating prediction (all scorable test pairs)", "", df_to_markdown(rating), "",
        "The hybrid ranks with a blended score, so it has no rating prediction of its own.", "",
        "## Is the difference real? Paired bootstrap on per-user NDCG", "", df_to_markdown(significance), "",
    ]
    if cold:
        lines += ["## Cold-start users", "",
                  f"{cold['cold_users_evaluated']:,} test users have no training history. All three models fall "
                  f"back to the popularity list for them: Precision@{k} = {cold[f'precision_at_{k}']:.4f}, "
                  f"Recall@{k} = {cold[f'recall_at_{k}']:.4f}, NDCG@{k} = {cold[f'ndcg_at_{k}']:.4f}.", ""]
    lines += ["## Example recommendations (top 5, * = the user rated it 4+ in the test period)", "",
              df_to_markdown(examples), ""]
    (out_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
