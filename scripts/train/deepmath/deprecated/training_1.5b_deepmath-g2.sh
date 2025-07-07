#!/bin/bash
set -x

export WANDB_MODE=offline
export WANDB_DIR=./outputs/wandb

N_GPUS=2
BASE_MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

TOTAL_EPOCHS=2 # 10000 * 2 / 128 = 156 steps
SAVE_STEPS=50
EVAL_STEPS=25
ROLLOUT_N=4
MAX_LENGTH=4096

EXPERIMENT_NAME="R1-Distill-Qwen-1.5B-deepmath-level5-6-beta-max-length-${MAX_LENGTH}-rollout-${ROLLOUT_N}"

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/deepmath_level5-6/train.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=512 \
 data.max_prompt_length=256 \
 data.max_response_length=$MAX_LENGTH \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=1.0 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl_rlvr'\
 trainer.username='aochongoliverli' \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 trainer.resume_mode='disable' \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_STEPS \
 trainer.test_freq=$EVAL_STEPS \
 trainer.total_epochs=$TOTAL_EPOCHS \
 trainer.push_to_hub=False 2>&1 | tee $CHECKPOINTS_DIR/verl_rlvr.log

 ## Push Saved Checkpoints to Huggingface
#  python3 push_to_hf/experiments.py --project_name verl_rlvr --run_name $EXPERIMENT_NAME