import numpy as np
import pytest

from src.evaluation import metrics as M


@pytest.fixture
def case():
    # 2 users, 6 items. User 0 likes items 1 and 4, user 1 likes item 5.
    relevant = np.zeros((2, 6), dtype=bool)
    relevant[0, [1, 4]] = True
    relevant[1, 5] = True
    topk = np.array([[1, 2, 4], [0, 1, 2]])
    return topk, relevant


def test_precision_recall_hit(case):
    topk, relevant = case
    hits = M.hits_matrix(topk, relevant)
    n_rel = relevant.sum(axis=1)
    assert M.precision_at_k(hits, 3).tolist() == pytest.approx([2 / 3, 0.0])
    assert M.recall_at_k(hits, n_rel, 3).tolist() == pytest.approx([1.0, 0.0])
    assert M.hit_rate_at_k(hits, 3).tolist() == [1.0, 0.0]


def test_ndcg_by_hand(case):
    topk, relevant = case
    hits = M.hits_matrix(topk, relevant)
    ndcg = M.ndcg_at_k(hits, relevant.sum(axis=1), 3)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)      # hits at rank 1 and 3
    idcg = 1 / np.log2(2) + 1 / np.log2(3)     # two hits at ranks 1 and 2
    assert ndcg[0] == pytest.approx(dcg / idcg)
    assert ndcg[1] == 0.0


def test_perfect_ranking_has_ndcg_one():
    relevant = np.array([[True, True, False, False]])
    hits = M.hits_matrix(np.array([[0, 1, 2]]), relevant)
    assert M.ndcg_at_k(hits, relevant.sum(axis=1), 3)[0] == pytest.approx(1.0)


def test_coverage_counts_distinct_items(case):
    topk, _ = case
    assert M.catalog_coverage(topk, 3, n_candidates=6) == pytest.approx(4 / 6)


def test_diversity_extremes():
    genres = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    same = np.array([[0, 1]])        # both pure genre A
    different = np.array([[0, 2]])   # A and B
    assert M.intra_list_diversity(same, 2, genres)[0] == pytest.approx(0.0)
    assert M.intra_list_diversity(different, 2, genres)[0] == pytest.approx(1.0)


def test_personalization():
    identical = np.array([[0, 1], [0, 1], [0, 1]])
    disjoint = np.array([[0, 1], [2, 3], [4, 5]])
    assert M.personalization(identical, 2, 6) == pytest.approx(0.0)
    assert M.personalization(disjoint, 2, 6) == pytest.approx(1.0)


def test_rating_errors():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.3, 0.1, 500)
    b = a - 0.05
    mean, lo, hi = M.bootstrap_diff_ci(a, b, n=200)
    assert mean == pytest.approx(0.05)
    assert lo <= 0.05 <= hi
