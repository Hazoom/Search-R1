import torch

from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto

PAD_ID = 0


class _FakeTokenizer:
    pad_token_id = PAD_ID


class _FakeManager:
    def __init__(self, pending_doc_len):
        self.tokenizer = _FakeTokenizer()
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=PAD_ID, max_prompt_length=50, max_obs_length=50, max_start_length=50
        ))
        self._traj_pending_doc_len = pending_doc_len
        self._purify_memorized_documents = LLMGenerationManager._purify_memorized_documents.__get__(self)


def test_purify_nulls_and_compacts_the_pending_document_span():
    mgr = _FakeManager(pending_doc_len=[3, 0])
    # row 0: [PAD PAD old(1,2,3) doc(6,7,8) memory_response(9)] -- doc span pending_doc_len=3
    # row 1: [PAD*5 old(1,2,3) response(4)] -- nothing pending
    rollings = DataProto.from_dict({
        'input_ids': torch.tensor([
            [0, 0, 1, 2, 3, 6, 7, 8, 9],
            [0, 0, 0, 0, 0, 1, 2, 3, 4],
        ]),
    })
    responses_ids = torch.tensor([[9], [4]])
    next_obs_ids = torch.tensor([[0], [0]])  # empty observation for both (memory action)
    is_memory = [1, 0]

    new_rollings = mgr._purify_memorized_documents(rollings, responses_ids, next_obs_ids, is_memory)

    # Row 0: the doc span [6,7,8] is gone; old content and the memory response compact together.
    assert new_rollings.batch['input_ids'][0].tolist() == [0, 0, 0, 0, 0, 1, 2, 3, 9]
    # Row 1: untouched, since it wasn't memorized and had nothing pending.
    assert new_rollings.batch['input_ids'][1].tolist() == [0, 0, 0, 0, 0, 1, 2, 3, 4]
    # Pending state is cleared for the purified row, untouched for the other.
    assert mgr._traj_pending_doc_len == [0, 0]


def test_purify_is_a_noop_when_no_row_memorized():
    mgr = _FakeManager(pending_doc_len=[3])
    rollings = DataProto.from_dict({'input_ids': torch.tensor([[0, 1, 2, 3]])})
    responses_ids = torch.tensor([[3]])
    next_obs_ids = torch.tensor([[0]])

    result = mgr._purify_memorized_documents(rollings, responses_ids, next_obs_ids, is_memory=[0])

    assert result is rollings  # short-circuits, returns the same object unmodified
    assert mgr._traj_pending_doc_len == [3]  # untouched


def test_purify_skips_rows_with_nothing_pending():
    mgr = _FakeManager(pending_doc_len=[0])
    rollings = DataProto.from_dict({'input_ids': torch.tensor([[0, 1, 2, 3]])})
    responses_ids = torch.tensor([[3]])
    next_obs_ids = torch.tensor([[0]])

    # is_memory=True but nothing was pending -- e.g. the model emitted a memory tag
    # without ever having searched. Should be a no-op, not an error.
    result = mgr._purify_memorized_documents(rollings, responses_ids, next_obs_ids, is_memory=[1])

    assert result is rollings
