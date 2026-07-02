from verl.utils.reward_score.se_search import compute_score_dense

GROUND_TRUTH = {'target': ['Paris']}

# nq_search.py's se_search prompt template embeds an example <answer> tag, so
# extract_solution requires >=2 <answer> matches before trusting the last one --
# mirror that here (matches qa_em.py's convention).
PROMPT_EXAMPLE = 'For example, <answer> Donald Trump </answer>. Question: What is the capital of France?'


def test_correct_answer_with_memory_and_one_search():
    solution = (
        PROMPT_EXAMPLE
        + ' <search> capital city of France </search> <memory>Paris is the capital</memory> <answer>Paris</answer>'
    )
    score = compute_score_dense(
        solution_str=solution,
        ground_truth=GROUND_TRUTH,
        search_queries=['capital city of France'],
        memory_text='Paris is the capital of France',
        format_violations=0,
        max_turns=5,
        alpha=0.1,
        gamma=0.01,
        mu=1.0,
    )
    # R_ans=1.0 (exact match F1), R_mem=1.0 (covered), R_query=-1 (1 atomic query,
    # correct answer -> discourage further search), R_format=0.
    # 1.0 + 0.1*1.0 + 0.01*1.0*(-1) + 0.01*0 = 1.09
    assert score == 1.09


def test_wrong_answer_gets_zero_f1_and_no_memory_credit():
    solution = PROMPT_EXAMPLE + ' <answer>Berlin</answer>'
    score = compute_score_dense(
        solution_str=solution,
        ground_truth=GROUND_TRUTH,
        search_queries=['capital city of France', 'population of France', 'history of Paris'],
        memory_text='',
        format_violations=1,
        max_turns=5,
        alpha=0.1,
        gamma=0.01,
        mu=1.0,
    )
    # R_ans=0 (no token overlap), R_mem=0 (empty memory), R_format=-1.
    # R_query = min(max_turns, n_query), and the formula must self-consistently
    # cancel the format penalty here: gamma*mu*n_query - gamma*1 == 0 => n_query == 1.
    assert score == 0.0


def test_no_answer_tag_scores_zero():
    solution = PROMPT_EXAMPLE + ' <search>capital of france</search>'
    score = compute_score_dense(
        solution_str=solution,
        ground_truth=GROUND_TRUTH,
        search_queries=[],
        memory_text='',
        format_violations=0,
        max_turns=5,
    )
    assert score == 0.0


def test_query_reward_discourages_extra_search_once_correct():
    solution = PROMPT_EXAMPLE + ' <answer>Paris</answer>'
    three_distinct_queries = [
        'who is the president of france in 2020',
        'Tencent YoutuLab GRPO cosine decay schedule',
        'penguins migration patterns in Antarctica region',
    ]
    many_queries = compute_score_dense(
        solution, GROUND_TRUTH, search_queries=three_distinct_queries, memory_text='',
        format_violations=0, max_turns=5, alpha=0.0, gamma=1.0, mu=1.0,
    )
    # With more valid atomic queries but a correct answer, R_query = -max(1, n) which
    # gets more negative as n grows -- more search after already being correct is
    # penalized harder.
    one_query = compute_score_dense(
        solution, GROUND_TRUTH, search_queries=['a single distinct query text here'],
        memory_text='', format_violations=0, max_turns=5, alpha=0.0, gamma=1.0, mu=1.0,
    )
    assert many_queries < one_query
