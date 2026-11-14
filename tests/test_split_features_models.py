import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.features.build import build_all
from src.features.store import FeatureContractError, FeatureStore
from src.preparation.clean import clean_items
from src.training.models import ALSMatrixFactorization, PopularityRecommender
from src.transformation.transform import build_matrix, time_split

FCFG = {"bayesian_prior_weight": 5, "trending_window_days": 30, "recency_half_life_days": 90,
        "positive_rating_threshold": 4}


@pytest.fixture
def toy():
    ts = pd.date_range("2000-06-01", periods=10, freq="7D", tz="UTC")
    inter = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3, 1, 2, 3, 1], "item_id": [10, 11, 10, 12, 11, 12, 12, 11, 10, 13],
                          "rating": [5, 3, 4, 2, 5, 4, 1, 5, 3, 4], "event_time": ts})
    # resolution-independent epoch seconds (pandas 3 defaults to microseconds)
    inter["timestamp"] = (inter["event_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    users = pd.DataFrame({"user_id": [1, 2, 3], "gender": ["M", "F", "U"], "age_group": ["25-34"] * 3,
                          "occupation": ["writer"] * 3})
    items = pd.DataFrame({"item_id": [10, 11, 12, 13], "release_year": [1995, 1990, 1999, 1980],
                          "genres": ["Comedy", "Drama|Comedy", "Action", ""], "n_genres": [1, 2, 1, 0]})
    return inter, users, items


def test_time_split_has_no_overlap(toy):
    inter, _, _ = toy
    train, test, cutoff = time_split(inter, 0.3)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert (train["event_time"] <= cutoff).all() and (test["event_time"] > cutoff).all()
    assert len(train) + len(test) == len(inter)


def test_features_ignore_events_after_reference_time(toy):
    inter, users, items = toy
    ref = inter["event_time"].iloc[5]
    feats = build_all(inter, users, items, ref, FCFG)
    # only the first 6 events count; item 13 is rated after ref so has zero popularity
    itf = feats["item_features"].set_index("item_id")
    assert itf.loc[13, "item_popularity"] == 0
    assert feats["interaction_features"]["event_time"].max() <= ref
    assert feats["user_features"].set_index("user_id").loc[1, "user_activity_count"] == 2


def test_feature_contract_rejects_unregistered_columns(toy, tmp_path, monkeypatch):
    inter, users, items = toy
    feats = build_all(inter, users, items, inter["event_time"].max(), FCFG)

    class Cfg(dict):
        def path(self, key, *parts):
            return tmp_path

    store = FeatureStore(Cfg())
    store.check_contract(feats)  # the real output satisfies the registry
    feats["user_features"]["sneaky_feature"] = 1.0
    with pytest.raises(FeatureContractError):
        store.check_contract(feats)


def test_article_moved_and_year_parsed():
    raw = pd.DataFrame({"item_id": ["1", "2"], "title": ["Matrix, The (1999)", "Promise, The (La Promesse) (1996)"],
                        "genres": ["Action|Sci-Fi", "drama"]})
    out = clean_items(raw, {})
    assert out["title"].tolist() == ["The Matrix", "The Promise (La Promesse)"]
    assert out["release_year"].tolist() == [1999, 1996]
    assert out["genres"].tolist() == ["Action|Sci-Fi", "Drama"]


def test_popularity_ranks_by_count_and_mf_learns(toy):
    inter, _, _ = toy
    m, uids, iids = build_matrix(inter)
    feats = pd.DataFrame({"item_id": iids, "item_popularity": [3, 3, 3, 1], "item_bayesian_rating": [4.0, 4.2, 2.5, 4.0]})
    pop = PopularityRecommender().fit(feats, uids, iids)
    assert np.argmax(pop.score(np.array([0]))[0]) == 1        # tie on count, broken by rating

    rng = np.random.default_rng(0)
    big = sp.random(60, 40, density=0.3, random_state=1, format="csr")
    big.data = rng.integers(1, 6, big.nnz).astype(float)
    mf = ALSMatrixFactorization(factors=4, reg=0.05, iterations=6).fit(big, np.arange(60), np.arange(40))
    assert mf.history[-1]["train_rmse"] < mf.history[0]["train_rmse"] + 1e-9
    preds = mf.predict_rating(*big.nonzero())
    assert preds.min() >= 1 and preds.max() <= 5
