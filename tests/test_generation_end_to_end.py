"""End-to-end test of LLMGenerationManager.run_llm_loop driving a scripted
search -> memory -> answer trajectory through the real rollout loop, actor
generation and the retrieval HTTP call are the only things faked (see
ScriptedActorRolloutWG / _fake_retrieve_post below) -- everything else (tokenization,
context purification, loss weighting, reward computation) runs for real.
"""
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from verl import DataProto
from verl.utils.reward_score.se_search import compute_score_dense

SCRIPTED_TURNS = [
    "<think>I need to find the capital of France.</think><search>capital of France</search>",
    "<think>The documents say Paris is the capital.</think><memory>Paris is the capital of France</memory>",
    "<think>I now know the answer.</think><answer>Paris</answer>",
]

RETRIEVED_PASSAGE = '"France"\nParis is the capital and most populous city of France.'


class ScriptedActorRolloutWG:
    """Stands in for the real Ray/FSDP/vLLM actor_rollout_wg. Returns one scripted
    response per call and records the exact input context it was given, so tests can
    verify the rolling context was purified correctly between turns."""

    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = turns
        self.call_count = 0
        self.seen_inputs = []

    def generate_sequences(self, active_batch: DataProto) -> DataProto:
        self.seen_inputs.append(
            self.tokenizer.batch_decode(active_batch.batch['input_ids'], skip_special_tokens=True)
        )
        text = self.turns[self.call_count]
        self.call_count += 1
        response_ids = self.tokenizer(
            [text], add_special_tokens=False, return_tensors='pt', padding='longest'
        )['input_ids']
        return DataProto.from_dict({'responses': response_ids}, meta_info={})


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _fake_retrieve_post(url, json):
    n = len(json['queries'])
    doc = {"document": {"contents": RETRIEVED_PASSAGE}, "score": 0.99}
    return _FakeHTTPResponse({"result": [[doc] for _ in range(n)]})


@pytest.fixture(scope='module')
def tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def final_output_and_actor(tokenizer):
    # Mirrors the real se_search prompt template (scripts/data_process/nq_search.py):
    # it embeds an example <answer> tag, which qa_em.extract_solution relies on --
    # it only trusts the *last* of >=2 <answer> matches as the model's real answer,
    # precisely so the prompt's own example doesn't get mistaken for the answer.
    prompt_text = (
        "You are a capable reasoning assistant... For example, <answer> Donald Trump </answer>. "
        "Question: What is the capital of France?"
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt')['input_ids']

    config = GenerationConfig(
        max_turns=3, max_start_length=512, max_prompt_length=1024, max_response_length=128,
        max_obs_length=128, num_gpus=1, search_url='http://fake/retrieve', topk=1,
        alpha=0.1, gamma=0.01,
    )
    actor_wg = ScriptedActorRolloutWG(tokenizer, SCRIPTED_TURNS)
    mgr = LLMGenerationManager(tokenizer=tokenizer, actor_rollout_wg=actor_wg, config=config)

    gen_batch = DataProto.from_dict({
        'input_ids': prompt_ids,
        'attention_mask': torch.ones_like(prompt_ids),
        'position_ids': torch.arange(prompt_ids.shape[1]).unsqueeze(0),
    })

    with patch('search_r1.llm_agent.generation.requests.post', side_effect=_fake_retrieve_post):
        final_output = mgr.run_llm_loop(gen_batch=gen_batch, initial_input_ids=prompt_ids)

    return final_output, actor_wg, mgr


def test_context_purification_removes_raw_documents_but_keeps_memory(final_output_and_actor):
    _, actor_wg, _ = final_output_and_actor
    turn3_input = actor_wg.seen_inputs[2][0]
    assert 'most populous city' not in turn3_input  # raw retrieved passage is gone
    assert 'Paris is the capital of France' in turn3_input  # distilled memory persists
    assert 'capital of France' in turn3_input  # the model's own search-tag text is untouched


def test_pending_doc_len_is_cleared_after_memorization(final_output_and_actor):
    _, _, mgr = final_output_and_actor
    assert mgr._traj_pending_doc_len == [0]


def test_trajectory_non_tensor_batch_is_populated(final_output_and_actor):
    final_output, _, _ = final_output_and_actor
    assert list(final_output.non_tensor_batch['search_queries'][0]) == ['capital of France']
    assert final_output.non_tensor_batch['memory_text'][0] == 'Paris is the capital of France'
    assert int(final_output.non_tensor_batch['format_violations'][0]) == 0


def test_loss_weight_tensor_is_well_formed(final_output_and_actor):
    final_output, _, _ = final_output_and_actor
    response_len = final_output.batch['responses'].shape[-1]
    loss_weight = final_output.batch['loss_weight'][:, -response_len:]
    assert loss_weight.min() >= 0.0
    assert loss_weight.max() <= 1.0
    assert (loss_weight == 0.01).any()  # the search query span
    assert (loss_weight == 0.1).any()  # the memory span


def test_dense_reward_matches_hand_computed_value(final_output_and_actor, tokenizer):
    final_output, _, _ = final_output_and_actor
    full_ids = final_output.batch['input_ids'][0]
    solution_str = tokenizer.decode(full_ids)

    score = compute_score_dense(
        solution_str=solution_str,
        ground_truth={'target': ['Paris']},
        search_queries=list(final_output.non_tensor_batch['search_queries'][0]),
        memory_text=str(final_output.non_tensor_batch['memory_text'][0]),
        format_violations=int(final_output.non_tensor_batch['format_violations'][0]),
        max_turns=3,
        alpha=0.1,
        gamma=0.01,
        mu=1.0,
    )
    # R_ans=1.0 (correct), R_mem=1.0 (memory covers the answer), R_query=-1 (1 atomic
    # query, correct answer), R_format=0: 1.0 + 0.1*1.0 + 0.01*1.0*(-1) + 0.01*0 = 1.09
    assert score == pytest.approx(1.09)
