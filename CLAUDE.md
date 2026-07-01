# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Search-R1 trains LLMs to interleave reasoning and search-engine tool calls using RL (PPO/GRPO/reinforce). It is built as a thin agentic layer (`search_r1/`) on top of a vendored, modified copy of **veRL** (`verl/`) — the installed Python package is actually named `verl` (see `pyproject.toml`/`setup.py`), and `search_r1` rides along as a sibling package.

There are two conda environments used in practice: `searchr1` (training: torch, vllm, veRL, flash-attn) and `retriever` (optional, for local retrieval servers: faiss-gpu, pyserini, fastapi/uvicorn).

## Common commands

```bash
# Install (searchr1 env)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip install -e .
pip3 install flash-attn --no-build-isolation

# Download indices/corpus and process a dataset
python scripts/download.py --save_path $save_path
python scripts/data_process/nq_search.py

# Launch the local retrieval server (retriever env) — must be running before training/inference
bash retrieval_launch.sh                          # e5 dense retriever, flat FAISS index, GPU
bash example/retriever/retrieval_launch_bm25.sh    # sparse BM25 alternative
bash example/retriever/retrieval_launch_serpapi.sh # online search alternative

# Train (searchr1 env), from repo root, with the retrieval server already running
bash train_ppo.sh
bash train_grpo.sh

# Inference against a trained checkpoint (retrieval server must be running)
python infer.py   # edit the `question` variable near the top of the file

# Build a custom retriever index
bash search_r1/search/build_index.sh
```

There is **no test suite** in this repo (`pyproject.toml` lists `pytest` as an optional dep but nothing uses it, and there is no `tests/` directory). Don't invent test commands; validate changes by running the relevant script/server directly.

Training is invoked as a Hydra app: `python3 -m verl.trainer.main_ppo <key.path>=<value> ...`. Config overrides are passed as `key.path=value` args (Hydra dot-notation), as seen throughout `train_ppo.sh`/`train_grpo.sh`. The base schema lives in `verl/trainer/config/ppo_trainer.yaml`.

## Architecture

### The core idea: an agentic rollout loop bolted onto veRL's PPO trainer

Vanilla veRL's PPO trainer generates one response per prompt per step. Search-R1's entire contribution is replacing that single-shot generation with a **multi-turn generate → parse → search → inject-observation → repeat** loop, plus a masking mechanism so the policy isn't trained to predict retrieved text.

- `verl/trainer/ppo/ray_trainer.py` — the Ray-based `RayPPOTrainer`. This is where Search-R1 hooks in: it imports `LLMGenerationManager`/`GenerationConfig` from `search_r1/llm_agent/generation.py` (this one import is essentially the entire diff from stock veRL). In `fit()`/`_validate()`, if `config.do_search` is true it calls `generation_manager.run_llm_loop(...)` instead of veRL's plain `actor_rollout_wg.generate_sequences(...)`.
- `search_r1/llm_agent/generation.py` — `LLMGenerationManager.run_llm_loop()` drives the loop, up to `max_turns` steps:
  1. Generate with the current policy (`actor_rollout_wg.generate_sequences`, GPU-padded via `_generate_with_gpu_padding`) only for still-active trajectories.
  2. Truncate each response at the first `</search>` or `</answer>` tag.
  3. Parse `<search>query</search>` or `<answer>...</answer>` via regex (`postprocess_predictions`). Batch all `search` queries into one HTTP POST to the retrieval server (`config.search_url`, i.e. `retriever.url` in the Hydra config); wrap the result as `\n\n<information>...</information>\n\n` and append it as the next observation. `answer` ends the trajectory; malformed output gets a corrective retry message instead of ending it.
  4. Track an `info_mask`/`responses_with_info_mask` alongside the normal token stream, so retrieved `<information>` text can be excluded from the PPO/GRPO loss later (`state_masking`, consumed in `verl/workers/actor/dp_actor.py` and `ray_trainer.py`'s `_create_loss_mask`).
  - `search_r1/llm_agent/tensor_helper.py` holds the padding/attention-mask/position-id plumbing used by the manager; no agent logic lives there.
- Search-R1-specific Hydra config keys (in `verl/trainer/config/ppo_trainer.yaml`, overridable from the training scripts): `retriever.url`, `retriever.topk`, `max_turns`, `do_search`, `algorithm.no_think_rl`, `algorithm.state_masking.{start,end}_state_marker`, `actor_rollout_ref.rollout.n_agent` (rollouts per prompt — 5 for GRPO group-relative advantage, 1 for PPO).

### Retrieval servers (`search_r1/search/`)

All retrievers expose the same FastAPI contract: `POST /retrieve` with `{queries, topk, return_scores}`, returning per-query ranked passages. This is what `LLMGenerationManager._batch_search` calls.

- `retrieval_server.py` — the main local server. `BM25Retriever` (pyserini/Lucene) or `DenseRetriever` (FAISS + HF encoder for e5/bge/DPR/T5-style models, optional `--faiss_gpu`), selected via `--retriever_name`.
- `serp_search_server.py` / `google_search_server.py` — online search alternatives (SerpAPI recommended over Google Custom Search — no monthly quota cap).
- `rerank_server.py` / `retrieval_rerank_server.py` — retrieval + reranker variants.
- `index_builder.py` (wrapped by `build_index.sh`) — builds the FAISS/BM25 index consumed by `retrieval_server.py` from a corpus jsonl.

Corpus format: one JSON object per line with `id` and `contents` (`"title"\ntext`) — see `example/corpus.jsonl`. Choice of retriever (sparse vs. dense-flat vs. dense-ANN vs. online) is a latency/accuracy/GPU tradeoff documented in `docs/retriever.md`.

### veRL internals relevant when touching training code

- `verl/trainer/main_ppo.py` is the Hydra entrypoint: builds tokenizer, worker classes (FSDP or Megatron strategy, dispatched from `config.actor_rollout_ref.actor.strategy`), a `RewardManager` (rule-based exact-match reward via `verl/utils/reward_score/qa_em.py`, dispatched by `data_source` — nq/triviaqa/popqa/hotpotqa/2wikimultihopqa/musique/bamboogle), and hands off to `RayPPOTrainer`. `main_ppo_format.py` is a variant using the format-aware reward (`qa_em_format.py`, used by v0.3 scripts).
- `verl/workers/rollout/` — generation backends: `vllm_rollout/` (used by all provided scripts, `rollout.name=vllm`) and `hf_rollout.py` (plain HF `generate`). No SGLang backend in this vendored copy.
- `verl/workers/actor/`, `verl/workers/critic/` — FSDP (`dp_*.py`) and Megatron (`megatron_*.py`) variants; GRPO runs critic-free.
- `verl/workers/sharding_manager/` — FSDP↔vLLM and Megatron↔vLLM weight resharding between training and rollout phases (veRL's "HybridEngine" mechanism), plus Ulysses sequence parallelism.
- When editing `verl/`, keep in mind this is a modified fork of the upstream `volcengine/verl` project (see `VERL_README.md` for the upstream project's own docs/links) — check whether a change belongs in the Search-R1-specific agent layer (`search_r1/`) vs. general veRL plumbing (`verl/`) before touching shared trainer/worker code.

### Data pipeline

Training/eval data is parquet with a fixed schema: `data_source`, `prompt` (chat-format list), `ability`, `reward_model.{style, ground_truth}`, `extra_info.{split, index}` (documented in the README's "Use your own dataset" section). `scripts/data_process/nq_search.py` is the canonical example — it builds prompts that instruct the model to reason in `<think>`, call `<search>query</search>`, and answer in `<answer>...</answer>`, and is the pattern to follow for new datasets/corpora. `scripts/nq_hotpotqa/v0.1/`, `v0.2/`, `v0.3/` are versioned snapshots of the full training recipe (data processing + train scripts + eval) as the project evolved — check the matching version's scripts when reproducing a specific paper result rather than assuming the root-level `train_ppo.sh`/`train_grpo.sh` match every version.

### Multi-node training

Ray-based: start a head node (`ray start --head`) and worker nodes (`ray start --address=...`), launch the **same retrieval server on every node**, then submit the job only from the head node via `ray job submit ... -- python3 -m verl.trainer.main_ppo ... trainer.nnodes=$N_NODES`. Details in `docs/multinode.md`.
