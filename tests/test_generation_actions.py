import torch

from search_r1.llm_agent.generation import LLMGenerationManager


class _FakeManager:
    """Binds just the LLMGenerationManager methods under test onto a bare object,
    avoiding the need for a real tokenizer/actor_rollout_wg/Ray worker group."""

    def __init__(self, batch_size):
        self._traj_search_queries = [[] for _ in range(batch_size)]
        self._traj_memory_text = [''] * batch_size
        self._traj_format_violations = [0] * batch_size
        self.postprocess_predictions = LLMGenerationManager.postprocess_predictions.__get__(self)
        self.execute_predictions = LLMGenerationManager.execute_predictions.__get__(self)
        self.batch_search = lambda queries: ['fake result'] * len(queries)


def test_postprocess_predictions_recognizes_all_three_actions():
    mgr = _FakeManager(batch_size=1)
    predictions = [
        '<search> capital of France </search>',
        '<memory> Paris is the capital </memory>',
        '<answer> Paris </answer>',
        '<garbled>not a real tag</garbled>',
    ]
    actions, contents = mgr.postprocess_predictions(predictions)
    assert actions == ['search', 'memory', 'answer', None]
    assert contents == ['capital of France', 'Paris is the capital', 'Paris', '']


def test_execute_predictions_handles_search_memory_and_invalid_actions():
    mgr = _FakeManager(batch_size=3)
    predictions = [
        '<search> capital of France </search>',
        '<memory> Paris is the capital </memory>',
        '<garbled>not a real tag</garbled>',
    ]
    active_mask = torch.tensor([True, True, True])
    next_obs, dones, valid_action, is_search, is_memory = mgr.execute_predictions(
        predictions, '<pad>', active_mask, do_search=True
    )

    assert '<information>fake result</information>' in next_obs[0]
    assert next_obs[1] == ''  # memory injects no observation
    assert 'previous action is invalid' in next_obs[2]

    assert dones == [0, 0, 0]  # search/memory continue; invalid retries -- none terminal
    assert valid_action == [1, 1, 0]
    assert is_search == [1, 0, 0]
    assert is_memory == [0, 1, 0]

    assert mgr._traj_search_queries == [['capital of France'], [], []]
    assert mgr._traj_memory_text == ['', 'Paris is the capital', '']
    assert mgr._traj_format_violations == [0, 0, 1]


def test_execute_predictions_answer_action_is_terminal():
    mgr = _FakeManager(batch_size=1)
    active_mask = torch.tensor([True])
    next_obs, dones, valid_action, is_search, is_memory = mgr.execute_predictions(
        ['<answer>Paris</answer>'], '<pad>', active_mask, do_search=True
    )
    assert dones == [1]
    assert valid_action == [1]
    assert is_search == [0]
    assert is_memory == [0]


def test_execute_predictions_skips_inactive_rows():
    mgr = _FakeManager(batch_size=2)
    active_mask = torch.tensor([True, False])
    predictions = ['<search>capital of France</search>', '<answer>anything</answer>']
    next_obs, dones, valid_action, is_search, is_memory = mgr.execute_predictions(
        predictions, '<pad>', active_mask, do_search=True
    )
    # Row 1 is inactive: no search call issued for it, treated as already-done.
    assert dones[1] == 1
    assert valid_action[1] == 0
    assert mgr._traj_search_queries[1] == []
