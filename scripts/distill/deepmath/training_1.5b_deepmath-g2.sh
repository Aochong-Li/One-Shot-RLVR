#!/bin/bash
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path
# Dataset Size: 56445
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
export MODEL_PATH="aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_1800"

N_GPUS=2
EXPERIMENT_NAME="Qwen2.5-Math-1.5B-deepmath-hard-1800-steps-4096"
ROLLOUT_N=2

python3 -m verl.trainer.main_distill_data \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/deepmath_4096_hard/train.parquet \
 data.train_batch_size=128 \
 data.max_prompt_length=256 \
 data.max_response_length=3840 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 trainer.logger=['console'] \
 trainer.project_name='verl_rlvr'\
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.resume_mode='disable' \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.push_to_hub=False 2>&1 | tee verl_demo.log