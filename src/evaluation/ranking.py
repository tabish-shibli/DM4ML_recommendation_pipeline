"""Top-K generation and the evaluation protocol, shared by tuning and testing.

Protocol
  - Candidates: movies with at least one rating in the training window.
    A model cannot sensibly recommend something nobody has rated yet.
  - Movies the user already rated in training are never recommended again.
  - Evaluated users: users seen in training who rated at least one movie
    >= relevance_threshold in the holdout period ("warm" users).
  - Relevant items: those holdout movies rated >= relevance_threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation import metrics as M


@dataclass
class EvalSet:
    user_ids: np.ndarray          # evaluated users (ids)
    user_rows: np.ndarray         # their rows in the model's user index
    relevant: np.ndarray          # bool, users x items
    n_relevant: np.ndarray
    rating_rows: np.ndarray       # every holdout (user, item) pair the model can score, for RMSE/MAE
    rating_cols: np.ndarray
    rating_true: np.ndarray


def build_eval_set(holdout: pd.DataFrame, user_ids: np.ndarray, item_ids: np.ndarray, threshold: int) -> EvalSet:
    u_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    known = holdout[holdout["user_id"].isin(u_pos.index) & holdout["item_id"].isin(i_pos.index)]
    rows = u_pos.loc[known["user_id"]].to_numpy()
    cols = i_pos.loc[known["item_id"]].to_numpy()

    pos = known["rating"].to_numpy() >= threshold
    eval_users = np.unique(rows[pos])
    remap = pd.Series(np.arange(len(eval_users)), index=eval_users)
    relevant = np.zeros((len(eval_users), len(item_ids)), dtype=bool)
    relevant[remap.loc[rows[pos]].to_numpy(), cols[pos]] = True
    return EvalSet(user_ids=user_ids[eval_users], user_rows=eval_users, relevant=relevant,
                   n_relevant=relevant.sum(axis=1), rating_rows=rows, rating_cols=cols,
                   rating_true=known["rating"].to_numpy(dtype=float))


def top_k(model, user_rows: np.ndarray, seen: sp.csr_matrix, candidates: np.ndarray, k: int,
          batch: int = 1024) -> np.ndarray:
    out = np.empty((len(user_rows), k), dtype=np.int64)
    for start in range(0, len(user_rows), batch):
        rows = user_rows[start:start + batch]
        s = np.array(model.score(rows), dtype=float)
        s[:, ~candidates] = -np.inf
        s[seen[rows].toarray() > 0] = -np.inf
        part = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1, kind="stable")
        out[start:start + len(rows)] = np.take_along_axis(part, order, axis=1)
    return out


def ranking_metrics(topk: np.ndarray, es: EvalSet, k_values, genre_vectors: np.ndarray,
                    item_user_share: np.ndarray, n_candidates: int) -> tuple[dict, pd.DataFrame]:
    hits = M.hits_matrix(topk, es.relevant)
    summary, per_user = {}, {"user_id": es.user_ids}
    for k in k_values:
        p, r = M.precision_at_k(hits, k), M.recall_at_k(hits, es.n_relevant, k)
        n, h = M.ndcg_at_k(hits, es.n_relevant, k), M.hit_rate_at_k(hits, k)
        d, nov = M.intra_list_diversity(topk, k, genre_vectors), M.novelty(topk, k, item_user_share)
        summary.update({f"precision_at_{k}": p.mean(), f"recall_at_{k}": r.mean(), f"ndcg_at_{k}": n.mean(),
                        f"hit_rate_at_{k}": h.mean(), f"coverage_at_{k}": M.catalog_coverage(topk, k, n_candidates),
                        f"diversity_at_{k}": d.mean(), f"novelty_at_{k}": nov.mean(),
                        f"personalization_at_{k}": M.personalization(topk, k, es.relevant.shape[1])})
        per_user.update({f"precision_at_{k}": p, f"recall_at_{k}": r, f"ndcg_at_{k}": n})
    summary = {k: float(v) for k, v in summary.items()}
    return summary, pd.DataFrame(per_user)


def rating_metrics(model, es: EvalSet) -> dict:
    if not getattr(model, "predicts_ratings", False):
        return {}
    pred = model.predict_rating(es.rating_rows, es.rating_cols)
    return {"rmse": M.rmse(es.rating_true, pred), "mae": M.mae(es.rating_true, pred),
            "rated_pairs_scored": int(len(pred))}
