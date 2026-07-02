import torch
import numpy as np
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # SE-Search dense-reward/weighted-loss hyperparameters (Eq. 6, 12); unused unless
    # algorithm.dense_reward.enable is set.
    alpha: float = 0.1
    gamma: float = 0.01

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        def _truncate_at_first_action_tag(resp: str) -> str:
            for tag in ('</search>', '</memory>', '</answer>'):
                if tag in resp:
                    return resp.split(tag)[0] + tag
            return resp

        responses_str = [_truncate_at_first_action_tag(resp) for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        # When every observation this turn is the empty string (e.g. all active rows took
        # a `memory` action), the tokenizer returns a 0-width tensor that defaults to
        # float32, which then silently upcasts the whole rolling context to float via
        # torch.cat in _update_rolling_state. Token ids must stay integral.
        return next_obs_ids.long()

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _purify_memorized_documents(self, rollings: DataProto, responses_ids: torch.Tensor,
                                   next_obs_ids: torch.Tensor, is_memory: List[int]) -> DataProto:
        """Once a `memory` action distills a search's raw <information> block, null out
        that block's tokens in the rolling context (fed to future generation) so it
        scrolls out of the length budget instead of permanently occupying it. Reuses
        TensorHelper's existing pad-and-resort machinery: overwriting a span with
        pad_token_id and re-sorting pushes it left, where cut_to_effective_len's
        right-anchored truncation will drop it once it exceeds max_prompt_length.

        Only `rollings` is touched here -- the full trajectory log used for training
        (`original_right_side`/info_mask) is left untouched, since PPO must train on
        log-probs computed against the context the model actually saw at generation time.
        """
        if not any(is_memory):
            return rollings

        pad_id = self.tokenizer.pad_token_id
        input_ids = rollings.batch['input_ids'].clone()
        resp_real_len = (responses_ids != pad_id).sum(dim=1)
        obs_real_len = (next_obs_ids != pad_id).sum(dim=1)

        changed = False
        for i, memorized in enumerate(is_memory):
            doc_len = self._traj_pending_doc_len[i]
            if not memorized or doc_len == 0:
                continue
            tail = int(resp_real_len[i] + obs_real_len[i])
            end = -tail if tail > 0 else None
            input_ids[i, -(doc_len + tail):end] = pad_id
            self._traj_pending_doc_len[i] = 0
            changed = True

        if not changed:
            return rollings

        input_ids, _ = self.tensor_fn.convert_pad_structure(input_ids)
        attention_mask = self.tensor_fn.create_attention_mask(input_ids)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        new_rollings = DataProto.from_dict({
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        })
        new_rollings.meta_info.update(rollings.meta_info)
        return new_rollings

    def _compute_response_weights(self, responses_ids: torch.Tensor, responses_str: List[str]) -> torch.Tensor:
        """Per-token loss weight for this turn's response (SE-Search Eq. 6): 1.0 by
        default, `self.config.gamma` for the query text inside <search>...</search>,
        `self.config.alpha` for the memory text inside <memory>...</memory>, 0.0 on
        padding. Falls back to uniform weight=1.0 when the tokenizer isn't "fast" (no
        character-to-token offset mapping available) -- a safe degrade rather than a
        crash or a mis-weighted span.
        """
        pad_id = self.tokenizer.pad_token_id
        weights = (responses_ids != pad_id).float()

        if not getattr(self.tokenizer, 'is_fast', False):
            return weights

        cur_actions, contents = self.postprocess_predictions(responses_str)
        for i, (action, content) in enumerate(zip(cur_actions, contents)):
            if action not in ('search', 'memory') or not content:
                continue
            resp = responses_str[i]
            # Re-match (rather than resp.find(content)) to get the *tagged* occurrence's
            # span specifically -- a model's <think> block restating its own query text
            # verbatim would otherwise make .find() latch onto the wrong occurrence.
            tag_match = re.search(r'<(search|memory|answer)>(.*?)</\1>', resp, re.DOTALL)
            if not tag_match:
                continue
            start_char, end_char = tag_match.span(2)

            offsets = self.tokenizer(resp, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
            real_positions = (responses_ids[i] != pad_id).nonzero(as_tuple=True)[0]
            if len(real_positions) != len(offsets):
                # Tokenizing this string alone vs. as part of the batch disagreed on token
                # count -- stay safe rather than risk misaligned indexing.
                continue

            span_weight = self.config.gamma if action == 'search' else self.config.alpha
            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_end > tok_start and tok_start >= start_char and tok_end <= end_char:
                    weights[i, real_positions[tok_idx]] = span_weight

        return weights

    def _info_masked_concatenate_with_padding(self,
                prompt: torch.Tensor,
                prompt_with_mask: torch.Tensor,
                response: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True,
                prompt_weight: torch.Tensor = None,
                response_weight: torch.Tensor = None,
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to
        cover the information block if it exists. When prompt_weight/response_weight are given
        (SE-Search's continuous loss weight, Eq. 6), reorders a third parallel tensor through the
        exact same permutation -- 0.0 for the information block, matching info_mask semantics.
        """
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        track_weight = prompt_weight is not None
        if track_weight:
            tensors_with_weight = [prompt_weight, response_weight]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            if track_weight:
                tensors_with_weight.append(torch.zeros(info.size(), dtype=torch.float, device=info.device))

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        if track_weight:
            concatenated_weight = torch.cat(tensors_with_weight, dim=1)
            padded_weight = concatenated_weight.gather(1, sorted_indices)
            return padded_tensor, padded_tensor_with_info, padded_weight

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict,
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          cur_response_weights: torch.Tensor = None) -> Dict:
        """Update right side state."""
        weight_kwargs = {}
        if cur_response_weights is not None:
            weight_kwargs = {
                'prompt_weight': right_side['responses_loss_weight'],
                'response_weight': cur_response_weights,
            }

        if next_obs_ids != None:
            result = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids,
                    pad_to_left=False,
                    **weight_kwargs,
                )
        else:
            result = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False,
                    **weight_kwargs,
                )

        if cur_response_weights is not None:
            responses, responses_with_info_mask, responses_loss_weight = result
        else:
            responses, responses_with_info_mask = result

        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        out = {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
        if cur_response_weights is not None:
            out['responses_loss_weight'] = responses_loss_weight[:, :max_len]
        return out

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []],
            'responses_loss_weight': initial_input_ids[:, []].float(),
        }

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # SE-Search trajectory state, tracked per example across turns (see execute_predictions)
        self._traj_search_queries: List[List[str]] = [[] for _ in range(batch_size)]
        self._traj_memory_text: List[str] = [''] * batch_size
        self._traj_format_violations: List[int] = [0] * batch_size
        # Token length of the most recent not-yet-memorized search observation, per example
        # (0 = nothing pending). Consumed by _purify_memorized_documents.
        self._traj_pending_doc_len: List[int] = [0] * batch_size

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, is_memory = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            # Track the most recent not-yet-memorized search observation's real token length,
            # per example, so a later `memory` action knows exactly what span to purify.
            pad_id = self.tokenizer.pad_token_id
            obs_real_len = (next_obs_ids != pad_id).sum(dim=1)
            for i, searched in enumerate(is_search):
                if searched:
                    self._traj_pending_doc_len[i] = int(obs_real_len[i])

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            rollings = self._purify_memorized_documents(
                rollings, responses_ids, next_obs_ids, is_memory
            )
            cur_response_weights = self._compute_response_weights(responses_ids, responses_str)
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                cur_response_weights
            )

        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            active_before_closure = active_mask.clone()
            _, dones, valid_action, is_search, _ = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            # Trajectories still active going into forced closure that didn't cleanly
            # produce an answer ran out of turns -- count as a format violation (SE-Search R_format).
            for i, done in enumerate(dones):
                if active_before_closure[i] and not done:
                    self._traj_format_violations[i] += 1

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            cur_response_weights = self._compute_response_weights(responses_ids, responses_str)
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                cur_response_weights=cur_response_weights,
            )

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # SE-Search per-example trajectory artifacts, consumed by the dense reward function.
        # These MUST be attached as non_tensor_batch (not meta_info): meta_info is a batch-level
        # dict that DataProto.reorder()/repeat() do not keep aligned per-example, whereas
        # non_tensor_batch is sliced/reordered/repeated in lockstep with the tensor batch.
        # NOTE: built via np.empty + per-index assignment rather than np.array(nested_list,
        # dtype=object) -- if every trajectory happens to issue the same number of search
        # queries, the latter silently collapses to a 2D array instead of a 1D array-of-lists.
        search_queries_arr = np.empty(batch_size, dtype=object)
        for i, queries in enumerate(self._traj_search_queries):
            search_queries_arr[i] = queries
        traj_non_tensors = {
            'search_queries': search_queries_arr,
            'memory_text': np.array(self._traj_memory_text, dtype=object),
            'format_violations': np.array(self._traj_format_violations, dtype=object),
        }

        return self._compose_final_output(original_left_side, original_right_side, meta_info, traj_non_tensors)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            non_tensors: Dict = None) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        # SE-Search weighted loss objective (Eq. 6): continuous per-token weight, consumed
        # by ray_trainer.py's _create_loss_mask in place of the binary info_mask when
        # algorithm.dense_reward.enable is set. The prompt portion's value is irrelevant --
        # PPO never computes loss over the prompt -- only its shape matters here.
        final_output['loss_weight'] = torch.cat([
            torch.ones_like(left_side['input_ids'], dtype=torch.float),
            final_output['responses_loss_weight']
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)

        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search, is_memory = [], [], [], [], []

        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, content, active) in enumerate(zip(cur_actions, contents, active_mask)):

            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                is_memory.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    is_memory.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    is_memory.append(0)
                    self._traj_search_queries[i].append(content)
                elif action == 'memory':
                    next_obs.append('')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                    is_memory.append(1)
                    self._traj_memory_text[i] = content
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to update my memory, I should put the content between <memory> and </memory>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    is_memory.append(0)
                    self._traj_format_violations[i] += 1

        assert len(search_results) == 0

        return next_obs, dones, valid_action, is_search, is_memory

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|memory|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
