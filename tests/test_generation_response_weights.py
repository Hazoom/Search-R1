import pytest
import torch
from transformers import AutoTokenizer

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig


@pytest.fixture(scope='module')
def gpt2_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class _FakeManager:
    def __init__(self, tokenizer, alpha=0.1, gamma=0.01):
        self.tokenizer = tokenizer
        self.config = GenerationConfig(
            max_turns=5, max_start_length=10, max_prompt_length=50, max_response_length=50,
            max_obs_length=50, num_gpus=1, alpha=alpha, gamma=gamma,
        )
        self.postprocess_predictions = LLMGenerationManager.postprocess_predictions.__get__(self)
        self._compute_response_weights = LLMGenerationManager._compute_response_weights.__get__(self)


def test_search_query_span_gets_gamma_weight(gpt2_tokenizer):
    mgr = _FakeManager(gpt2_tokenizer)
    resp = '<think>I need to search</think><search>capital of France</search>'
    batch = gpt2_tokenizer([resp], add_special_tokens=False, return_tensors='pt', padding='longest')
    weights = mgr._compute_response_weights(batch['input_ids'], [resp])

    tokens = gpt2_tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
    weight_by_token = dict(zip(range(len(tokens)), weights[0].tolist()))
    query_token_idxs = [i for i, t in enumerate(tokens) if t in ('capital', 'Ġof', 'ĠFrance')]
    assert query_token_idxs, 'expected to find the query tokens in the tokenized response'
    for i in query_token_idxs:
        assert weight_by_token[i] == pytest.approx(0.01)
    # Everything outside the query span (tags, think content) stays at the default weight.
    assert weight_by_token[0] == pytest.approx(1.0)


def test_memory_span_gets_alpha_weight_not_gamma(gpt2_tokenizer):
    mgr = _FakeManager(gpt2_tokenizer)
    resp = '<think>Let me remember</think><memory>Paris is the capital of France</memory>'
    batch = gpt2_tokenizer([resp], add_special_tokens=False, return_tensors='pt', padding='longest')
    weights = mgr._compute_response_weights(batch['input_ids'], [resp])

    tokens = gpt2_tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
    memory_token_idxs = [i for i, t in enumerate(tokens) if t in ('Paris', 'Ġis', 'Ġthe', 'Ġcapital', 'Ġof', 'ĠFrance')]
    for i in memory_token_idxs:
        assert weights[0, i].item() == pytest.approx(0.1)


def test_think_block_restating_query_does_not_steal_the_weight(gpt2_tokenizer):
    """Regression test: resp.find(content) would match the *first* occurrence of the
    query text, which can be inside the <think> block if the model restates its own
    query verbatim while reasoning -- the actual <search> tag's content must get the
    weight, not an earlier restatement of the same words."""
    mgr = _FakeManager(gpt2_tokenizer)
    resp = '<think>I need to find the capital of France.</think><search>capital of France</search>'
    batch = gpt2_tokenizer([resp], add_special_tokens=False, return_tensors='pt', padding='longest')
    weights = mgr._compute_response_weights(batch['input_ids'], [resp])
    tokens = gpt2_tokenizer.convert_ids_to_tokens(batch['input_ids'][0])

    search_tag_idx = tokens.index('search')  # the opening <search> tag's "search" token
    # All weight-0.01 tokens must come after the <search> tag, never before it.
    for i, w in enumerate(weights[0].tolist()):
        if w == pytest.approx(0.01):
            assert i > search_tag_idx, f'token {tokens[i]!r} at {i} got query weight but precedes <search>'


def test_padding_gets_zero_weight(gpt2_tokenizer):
    mgr = _FakeManager(gpt2_tokenizer)
    responses = ['<answer>Paris</answer>', '<answer>A much longer answer here</answer>']
    batch = gpt2_tokenizer(responses, add_special_tokens=False, return_tensors='pt', padding='longest')
    weights = mgr._compute_response_weights(batch['input_ids'], responses)
    pad_mask = batch['input_ids'] == gpt2_tokenizer.pad_token_id
    assert torch.all(weights[pad_mask] == 0.0)
    assert torch.all(weights[~pad_mask] > 0.0)


def test_falls_back_to_uniform_weight_when_tokenizer_is_not_fast():
    class SlowTokenizer:
        pad_token_id = 0
        is_fast = False

    mgr = _FakeManager(SlowTokenizer())
    responses_ids = torch.tensor([[1, 2, 3, 0, 0]])
    weights = mgr._compute_response_weights(responses_ids, ['<search>capital of France</search>'])
    assert weights.tolist() == [[1.0, 1.0, 1.0, 0.0, 0.0]]
