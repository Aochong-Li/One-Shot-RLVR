#!/bin/bash
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path
# Dataset Size: 490K
# export HUGGINGFACE_HUB_TOKEN="YOUR TOKEN HERE"

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
export BASE_MODEL="aochongoliverli/Qwen2.5-3B-sft-distill-countdown-level3-4-150"

N_GPUS=1
ROLLOUT_N=3
MAX_LENGTH=4096
TENSOR_MODEL_PARALLEL_SIZE=1
TOTAL_EPOCHS=1
SAVE_STEPS=100
EVAL_STEPS=30

EXPERIMENT_NAME="Qwen2.5-3B-sft-1epoch-countdown-level4-5-${TOTAL_EPOCHS}epochs-${ROLLOUT_N}rollouts-${MAX_LENGTH}max-length"

python3 -m verl.trainer.main_ppo_debug \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/countdown_level45_20k/train.parquet \
 data.val_files=data/train/countdown_level45_20k/test.parquet \
 data.train_batch_size=6 \
 data.val_batch_size=256 \
 data.max_prompt_length=256 \
 data.max_response_length=$MAX_LENGTH \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=1.0 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console'] \
 countdown.corrupt_seq=True \
 trainer.project_name='countdown'\
 trainer.username=aochongoliverli \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 trainer.resume_mode='disable' \
 +trainer.val_before_train=False \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_STEPS \
 trainer.test_freq=$EVAL_STEPS \
 trainer.total_epochs=$TOTAL_EPOCHS \
 trainer.push_to_hub=False 2>&1 | tee $CHECKPOINTS_DIR/$EXPERIMENT_NAME.log

 ## Push Saved Checkpoints to Huggingface
#  python3 push_to_hf/experiments.py --project_name countdown --run_name $EXPERIMENT_NAME