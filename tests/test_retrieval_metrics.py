
import pytest

from src.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    measure_latency,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(
        2 / 3
    )


def test_precision_at_k_all_relevant():
    assert precision_at_k(
        ["a", "b"], {"a", "b"}, 2
    ) == 1.0


def test_precision_at_k_no_results():
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_precision_at_k_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "d"}

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(
        2 / 3
    )


def test_recall_at_k_no_relevant_documents():
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(
        ["a", "b", "c"], {"c"}
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant_result():
    assert reciprocal_rank(["a", "b"], {"c"}) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(
        0.5
    )


def test_mean_reciprocal_rank_empty():
    assert mean_reciprocal_rank([]) == 0.0


def test_measure_latency():
    result, elapsed_ms = measure_latency(lambda: "done")

    assert result == "done"
    assert elapsed_ms >= 0
