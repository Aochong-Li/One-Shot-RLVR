#!/bin/bash
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs/${EXPERIMENT_NAME}"

N_GPUS=8
EXPERIMENT_NAME="Qwen2.5-Math-1.5B-dsr_sub"
TOTAL_EPOCHS=150 # 150 x 9 = 1350 steps; 1350 / 100 = 14 checkpoints
SAVE_STEPS=100 # Save 14 checkpoints
EVAL_STEPS=50 # 1350 / 50 = 27 times 

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/one_shot_rlvr/dsr_sub.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
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
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=8 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl_rlvr'\
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_STEPS \
 trainer.test_freq=$EVAL_STEPS \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee verl_demo.log