
from time import perf_counter


def precision_at_k(retrieved, relevant, k):
    """Calculate Precision@K.

    retrieved: ranked list of retrieved chunk IDs
    relevant: collection of known relevant chunk IDs
    k: number of top results to evaluate
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_set = set(relevant)
    relevant_count = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_set
    )

    return relevant_count / len(top_k)


def recall_at_k(retrieved, relevant, k):
    """Calculate Recall@K against the known relevant chunk set."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    retrieved_set = set(retrieved[:k])

    return len(retrieved_set & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved, relevant):
    """Return the reciprocal rank of the first relevant result."""
    relevant_set = set(relevant)

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rankings):
    """Calculate the mean of reciprocal ranks."""
    if not rankings:
        return 0.0

    return sum(rankings) / len(rankings)


def measure_latency(function, *args, **kwargs):
    """Return (function_result, elapsed_time_in_ms)."""
    start = perf_counter()
    result = function(*args, **kwargs)
    elapsed_ms = (perf_counter() - start) * 1000

    return result, elapsed_ms
