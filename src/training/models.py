"""The three recommenders.

All of them share one interface so that tuning, evaluation and serving can
treat them the same way:

    score(user_rows)              -> users x items array, higher = better
    predict_rating(rows, cols)    -> predicted star rating (where it applies)

Rows and columns are positions in the index arrays the model was fitted
with (`user_ids`, `item_ids`), which come from the transformation stage.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.build import genre_match


class Recommender:
    name = "base"
    predicts_ratings = False

    def __init__(self) -> None:
        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.fit_seconds: float | None = None

    def user_rows(self, user_ids) -> np.ndarray:
        """Row position of each id, -1 for users the model has never seen."""
        pos = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        return pos.reindex(np.asarray(user_ids)).fillna(-1).astype(int).to_numpy()

    def item_cols(self, item_ids) -> np.ndarray:
        pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        return pos.reindex(np.asarray(item_ids)).fillna(-1).astype(int).to_numpy()

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path) -> "Recommender":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _align(features: pd.DataFrame, item_ids: np.ndarray, col: str, fill: float = 0.0) -> np.ndarray:
    return features.set_index("item_id")[col].reindex(item_ids).fillna(fill).to_numpy(dtype=float)


class PopularityRecommender(Recommender):
    """Baseline: everyone gets the most-rated movies they have not seen yet.

    Ratings are predicted with the Bayesian-smoothed item mean, which is the
    natural "non-personalised" rating guess and gives the MF model something
    honest to beat on RMSE.
    """
    name = "popularity"
    predicts_ratings = True

    def fit(self, item_features: pd.DataFrame, user_ids: np.ndarray, item_ids: np.ndarray):
        t0 = time.perf_counter()
        self.user_ids, self.item_ids = user_ids, item_ids
        self.popularity = _align(item_features, item_ids, "item_popularity")
        self.rating = _align(item_features, item_ids, "item_bayesian_rating",
                             fill=float(item_features["item_bayesian_rating"].median()))
        # ties in count are broken by the smoothed rating
        self._score = self.popularity + 1e-3 * self.rating
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._score, (len(user_rows), len(self._score))).copy()

    def predict_rating(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.rating[cols]


class ALSMatrixFactorization(Recommender):
    """Biased matrix factorisation trained with alternating least squares.

        r_ui ~ mu + b_u + b_i + p_u . q_i

    With item factors fixed, each user's [p_u, b_u] is a small ridge regression,
    and vice versa. Appending a constant 1 to the fixed side lets the bias be
    solved together with the factors. All users (or items) are solved at once:
    the per-row Gram matrices come from one sparse product with the stacked
    outer products, then a batched np.linalg.solve.
    Regularisation is scaled by each row's rating count (weighted-lambda ALS).
    """
    name = "matrix_factorization"
    predicts_ratings = True

    def __init__(self, factors: int = 32, reg: float = 0.1, iterations: int = 10, seed: int = 42):
        super().__init__()
        self.factors, self.reg, self.iterations, self.seed = factors, reg, iterations, seed
        self.history: list[dict] = []

    def params(self) -> dict:
        return {"factors": self.factors, "reg": self.reg, "iterations": self.iterations, "seed": self.seed}

    def _solve(self, centred: sp.csr_matrix, fixed: np.ndarray, fixed_bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self.factors
        aug = np.hstack([fixed, np.ones((fixed.shape[0], 1))]).astype(np.float32)
        k = f + 1
        outer = (aug[:, :, None] * aug[:, None, :]).reshape(aug.shape[0], k * k)
        pattern = centred.copy()
        pattern.data = np.ones_like(pattern.data)
        gram = np.asarray(pattern @ outer, dtype=np.float64).reshape(-1, k, k)
        counts = np.diff(centred.indptr)
        target = centred.copy()
        target.data = target.data - fixed_bias[target.indices].astype(np.float32)
        rhs = np.asarray(target @ aug, dtype=np.float64)
        diag = np.arange(k)
        gram[:, diag, diag] += (self.reg * np.maximum(counts, 1))[:, None]
        sol = np.linalg.solve(gram, rhs[..., None])[..., 0]
        return sol[:, :f], sol[:, f]

    def fit(self, ratings: sp.csr_matrix, user_ids: np.ndarray, item_ids: np.ndarray, log_every: int = 0):
        t0 = time.perf_counter()
        self.user_ids, self.item_ids = user_ids, item_ids
        rng = np.random.default_rng(self.seed)
        n_users, n_items = ratings.shape
        self.mu = float(ratings.data.mean())
        centred = ratings.astype(np.float32).tocsr()
        centred.data = centred.data - np.float32(self.mu)
        centred_t = centred.T.tocsr()

        self.P = rng.normal(0, 0.1, (n_users, self.factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.factors))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        rows, cols = ratings.nonzero()
        for it in range(self.iterations):
            self.P, self.bu = self._solve(centred, self.Q, self.bi)
            self.Q, self.bi = self._solve(centred_t, self.P, self.bu)
            train_rmse = float(np.sqrt(np.mean((self.predict_rating(rows, cols) - ratings.data) ** 2)))
            self.history.append({"iteration": it + 1, "train_rmse": train_rmse})
        self.fit_seconds = time.perf_counter() - t0
        return self

    def raw_score(self, user_rows: np.ndarray) -> np.ndarray:
        return self.mu + self.bu[user_rows, None] + self.bi[None, :] + self.P[user_rows] @ self.Q.T

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return self.raw_score(user_rows)

    def predict_rating(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        pred = self.mu + self.bu[rows] + self.bi[cols] + np.einsum("ij,ij->i", self.P[rows], self.Q[cols])
        return np.clip(pred, 1.0, 5.0)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)


class HybridRecommender(Recommender):
    """MF preference, popularity prior and genre match, blended per user.

    Each signal is standardised across the candidate movies for that user, so
    the weights are comparable. Predicted ratings come from the MF component;
    the hybrid score itself is only used for ranking.
    """
    name = "hybrid"
    predicts_ratings = False

    def __init__(self, mf: ALSMatrixFactorization, mf_weight: float = 1.0, popularity_weight: float = 1.0,
                 genre_weight: float = 0.0):
        super().__init__()
        self.mf = mf
        self.mf_weight, self.popularity_weight, self.genre_weight = mf_weight, popularity_weight, genre_weight

    def params(self) -> dict:
        return {"mf_weight": self.mf_weight, "popularity_weight": self.popularity_weight,
                "genre_weight": self.genre_weight, **{f"mf_{k}": v for k, v in self.mf.params().items()}}

    def fit(self, item_features: pd.DataFrame, user_affinity: pd.DataFrame, item_genres: pd.DataFrame):
        t0 = time.perf_counter()
        self.user_ids, self.item_ids = self.mf.user_ids, self.mf.item_ids
        logpop = np.log1p(_align(item_features, self.item_ids, "item_popularity"))
        self.pop_z = (logpop - logpop.mean()) / logpop.std()
        aff = user_affinity.set_index("user_id").reindex(self.user_ids).fillna(0.0)
        self.affinity = aff.to_numpy(dtype=float)
        self.genre_flags = item_genres.set_index("item_id").reindex(self.item_ids).fillna(0).to_numpy(dtype=float)
        self.fit_seconds = self.mf.fit_seconds + (time.perf_counter() - t0)
        return self

    def with_weights(self, mf_weight: float, popularity_weight: float, genre_weight: float) -> "HybridRecommender":
        clone = HybridRecommender(self.mf, mf_weight, popularity_weight, genre_weight)
        clone.__dict__.update({k: v for k, v in self.__dict__.items()
                               if k in ("user_ids", "item_ids", "pop_z", "affinity", "genre_flags", "fit_seconds")})
        return clone

    def score(self, user_rows: np.ndarray, affinity: np.ndarray | None = None) -> np.ndarray:
        """Blend the three signals. At serving time `affinity` comes from the
        online feature store instead of the copy taken at training time."""
        affinity = self.affinity[user_rows] if affinity is None else np.atleast_2d(affinity)
        s = self.mf_weight * _zscore_rows(self.mf.raw_score(user_rows))
        s = s + self.popularity_weight * self.pop_z[None, :]
        if self.genre_weight:
            s = s + self.genre_weight * _zscore_rows(genre_match(affinity, self.genre_flags))
        return s

    def predict_rating(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.mf.predict_rating(rows, cols)
