from collections import Counter
from typing import List

from . import qa_em
from search_r1.llm_agent.atomic_query import count_atomic_queries


def _f1_score(prediction: str, golden_answers) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred_tokens = qa_em.normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for golden_answer in golden_answers:
        gold_tokens = qa_em.normalize_answer(golden_answer).split()
        if not gold_tokens:
            continue
        num_common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if num_common == 0:
            continue
        f1 = 2 * num_common / (len(pred_tokens) + len(gold_tokens))
        best_f1 = max(best_f1, f1)
    return best_f1


def compute_score_dense(
    solution_str: str,
    ground_truth: dict,
    search_queries: List[str],
    memory_text: str,
    format_violations: int,
    max_turns: int,
    alpha: float = 0.1,
    gamma: float = 0.01,
    mu: float = 1.0,
) -> float:
    """SE-Search dense reward (arXiv:2603.03293), Eq. 12:
    R_Dense = R_ans + alpha*R_mem + gamma*mu*R_query + gamma*R_format
    """
    golden_answers = ground_truth['target']
    answer = qa_em.extract_solution(solution_str=solution_str)

    if answer is None:
        return 0.

    is_correct = bool(qa_em.em_check(answer, golden_answers))
    r_ans = _f1_score(answer, golden_answers)

    r_mem = float(qa_em.subem_check(memory_text, golden_answers)) if memory_text else 0.

    n_query = count_atomic_queries(search_queries)
    r_query = -max(1, n_query) if is_correct else min(max_turns, n_query)

    r_format = -1. * format_violations

    return r_ans + alpha * r_mem + gamma * mu * r_query + gamma * r_format
