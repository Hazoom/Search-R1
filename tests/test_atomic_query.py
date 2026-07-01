from search_r1.llm_agent.atomic_query import count_atomic_queries


def test_too_short_queries_are_filtered():
    assert count_atomic_queries(['short']) == 0


def test_too_long_queries_are_filtered():
    assert count_atomic_queries(['x' * 130]) == 0


def test_empty_query_list():
    assert count_atomic_queries([]) == 0


def test_near_duplicate_query_rejected():
    q1 = 'who is the president of france'
    q2 = 'who is the president of france '  # trivially near-identical
    assert count_atomic_queries([q1, q2]) == 1


def test_genuinely_distinct_queries_both_counted():
    q1 = 'who is the president of france in 2020'
    q2 = 'Tencent YoutuLab GRPO cosine decay schedule'
    assert count_atomic_queries([q1, q2]) == 2


def test_similarity_threshold_is_configurable():
    q1 = 'who is the president of france'
    q2 = 'what is the capital of germany'
    # Character-level similarity between these is ~0.67 -- above the default
    # threshold (0.3) they're treated as near-duplicates.
    assert count_atomic_queries([q1, q2], similarity_threshold=0.3) == 1
    # A much looser threshold accepts both.
    assert count_atomic_queries([q1, q2], similarity_threshold=0.9) == 2
