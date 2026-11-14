"""Ranking and rating metrics. Plain numpy so they are easy to unit test.

Conventions
  topk      int array, users x K, item column indices in ranked order
  relevant  bool array, users x n_items, True where the held-out rating >= threshold
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def hits_matrix(topk: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    return np.take_along_axis(relevant, topk, axis=1)


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return hits[:, :k].sum(axis=1) / k


def recall_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    return hits[:, :k].sum(axis=1) / np.maximum(n_relevant, 1)


def hit_rate_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return (hits[:, :k].sum(axis=1) > 0).astype(float)


def ndcg_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Binary-relevance NDCG. The ideal list puts min(n_relevant, k) hits first."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits[:, :k] * discounts).sum(axis=1)
    ideal_counts = np.minimum(n_relevant, k)
    cum = np.concatenate([[0.0], np.cumsum(discounts)])
    idcg = cum[ideal_counts]
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def catalog_coverage(topk: np.ndarray, k: int, n_candidates: int) -> float:
    """Share of recommendable movies that appear in at least one user's top-k."""
    return len(np.unique(topk[:, :k])) / n_candidates


def intra_list_diversity(topk: np.ndarray, k: int, genre_vectors: np.ndarray) -> np.ndarray:
    """Mean pairwise genre dissimilarity (1 - cosine) inside each user's list.

    0 means every recommended movie has the same genre profile, 1 means no two
    share a genre. Movies with no genre tags count as dissimilar to everything.
    """
    norms = np.linalg.norm(genre_vectors, axis=1, keepdims=True)
    unit = np.divide(genre_vectors, norms, out=np.zeros_like(genre_vectors, dtype=float), where=norms > 0)
    v = unit[topk[:, :k]]                          # users x k x genres
    sim = np.einsum("ukg,ulg->ukl", v, v)
    off_diag = sim.sum(axis=(1, 2)) - np.einsum("ukk->u", sim)
    return 1.0 - off_diag / (k * (k - 1))


def novelty(topk: np.ndarray, k: int, item_user_share: np.ndarray) -> np.ndarray:
    """Mean self-information -log2(p) of recommended movies, p = share of users who rated it."""
    p = np.clip(item_user_share[topk[:, :k]], 1e-9, 1.0)
    return (-np.log2(p)).mean(axis=1)


def personalization(topk: np.ndarray, k: int, n_items: int) -> float:
    """1 - mean overlap between any two users' top-k lists.

    0 means everyone gets the same list, values near 1 mean lists rarely share
    a movie. Computed over all user pairs with one sparse product.
    """
    n = len(topk)
    if n < 2:
        return 0.0
    rows = np.repeat(np.arange(n), k)
    x = sp.csr_matrix((np.ones(n * k), (rows, topk[:, :k].ravel())), shape=(n, n_items))
    shared = (x @ x.T).toarray()
    mean_overlap = (shared.sum() - np.trace(shared)) / (n * (n - 1) * k)
    return float(1.0 - mean_overlap)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap CI for mean(a - b) over the same users."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)
