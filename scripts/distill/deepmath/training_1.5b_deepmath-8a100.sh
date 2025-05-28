#!/bin/bash
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path
# Dataset Size: 46600

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
export MODEL_PATH="aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_1800"

N_GPUS=8
EXPERIMENT_NAME="Qwen2.5-Math-1.5B-deepmath-hard-1800-steps-4096"
ROLLOUT_N=8

# HACK:
# 1. entropy coeff = 0.003
# 2. max_num_batched_tokens, max_num_seqs

python3 -m verl.trainer.main_distill_data \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/deepmath_4096_hard/train.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=512 \
 data.max_prompt_length=256 \
 data.max_response_length=3840 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size= 1\
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 trainer.logger=['console'] \
 trainer.project_name='verl_rlvr'\
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.resume_mode='disable' \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.username='aochongoliverli' \
 trainer.push_to_hub=False 2>&1 | tee verl_demo.log