
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Fraction of top-K positions occupied by relevant documents."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_ids[:k]
    relevant_retrieved = set(top_k_ids) & set(relevant_ids)

    return len(relevant_retrieved) / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Fraction of all relevant documents found in top K."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    top_k_ids = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    return len(top_k_ids & relevant_set) / len(relevant_set)


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Return 1 if at least one relevant document appears in top K."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    return float(
        any(doc_id in relevant_set for doc_id in retrieved_ids[:k])
    )


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Reciprocal rank of the first relevant document."""

    relevant_set = set(relevant_ids)

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:
    """Average reciprocal rank across all queries."""

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def precision_at_retrieved(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Precision using the actual number of retrieved documents."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k_ids = retrieved_ids[:k]

    if not top_k_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    relevant_retrieved = sum(
        doc_id in relevant_set for doc_id in top_k_ids
    )

    return relevant_retrieved / len(top_k_ids)
