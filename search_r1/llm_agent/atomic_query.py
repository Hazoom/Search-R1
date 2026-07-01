import difflib
from typing import List


def count_atomic_queries(
    queries: List[str],
    min_len: int = 20,
    max_len: int = 120,
    similarity_threshold: float = 0.3,
) -> int:
    """Algorithm 1 (Atomic Query Counting) from SE-Search (arXiv:2603.03293).

    Counts the number of "atomic" queries in a trajectory: queries within a
    length band that are not near-duplicates of any query already accepted.
    """
    valid_queries: List[str] = []
    for query in queries:
        if not (min_len <= len(query) <= max_len):
            continue
        # Eq. 14: ratio(qi, qj) = 2 * sum(matching_block_lengths) / (|qi| + |qj|),
        # which is exactly what SequenceMatcher.ratio() computes.
        is_near_duplicate = any(
            difflib.SequenceMatcher(None, query, accepted).ratio() > similarity_threshold
            for accepted in valid_queries
        )
        if not is_near_duplicate:
            valid_queries.append(query)

    return len(valid_queries)
