# SMOKE TEST ONLY — single 16GB V100.
# Goal: verify the SE-Search loop (generate -> search -> memory -> dense reward -> GRPO update)
# runs end to end. This does NOT reproduce the paper (0.5B model, batch 8, ~10 steps).
#
# Prereqs:
#   1) Retriever running on 127.0.0.1:8000 (bash retrieval_launch.sh)
#   2) Data built with the SE-Search template:
#        python scripts/data_process/nq_search.py --local_dir ./data/nq_search --template_type se_search
#   3) V100 is Volta (SM 7.0): flash-attn is NOT used here (use_remove_padding=False),
#      and vLLM runs in float16 (no bf16 on V100).

export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='data/nq_search'
export VLLM_ATTENTION_BACKEND=XFORMERS

WAND_PROJECT='Search-R1'
export BASE_MODEL='Qwen/Qwen2.5-0.5B'
export EXPERIMENT_NAME=nq-se-search-grpo-qwen2.5-0.5b-v100-smoketest

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.max_start_length=1024 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.dense_reward.enable=true \
    algorithm.dense_reward.alpha=0.1 \
    algorithm.dense_reward.gamma=0.01 \
    algorithm.dense_reward.decay_steps=150 \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1000 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
