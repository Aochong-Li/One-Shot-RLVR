#!/bin/bash
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path
# Dataset Size: 56445
N_GPUS=8
ROLLOUT_N=6
TENSOR_MODEL_PARALLEL_SIZE=1
EXPERIMENT_NAME="Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-${ROLLOUT_N}"
TOTAL_EPOCHS=5 # 46000 * 5 / 128 = 1750 steps
SAVE_STEPS=200 # 1750 / 200 = 8 checkpoints
EVAL_STEPS=50 # 1750 / 50 = 35 times

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"

# HACK:
# 1. entropy coeff = 0.003
# 2. max_num_batched_tokens, max_num_seqs

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/deepmath_4096_hard/train.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=512 \
 data.max_prompt_length=256 \
 data.max_response_length=3840 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
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
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
 actor_rollout_ref.rollout.max_num_batched_tokens=8388608 \
 actor_rollout_ref.rollout.max_num_seqs=2048 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl_rlvr'\
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 trainer.resume_mode='disable' \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_STEPS \
 trainer.test_freq=$EVAL_STEPS \
 trainer.total_epochs=$TOTAL_EPOCHS \
 trainer.push_to_hub=False 2>&1 | tee verl_demo.log

 ## Push Saved Checkpoints to Huggingface
 python3 push_to_hf/experiments.py --project_name verl_rlvr --run_name $EXPERIMENT_NAME