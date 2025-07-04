#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
export BASE_MODEL="aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch"

N_GPUS=4
ROLLOUT_N=4
MAX_LENGTH=8192
TENSOR_MODEL_PARALLEL_SIZE=1
TOTAL_EPOCHS=2
SAVE_STEPS=25
EVAL_STEPS=5

EXPERIMENT_NAME="Qwen2.5-3B-countdown-level-5-${TOTAL_EPOCHS}epochs-${ROLLOUT_N}rollouts-${MAX_LENGTH}max-length-stage2-corrupt_numbers+original"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/countdown_level5_corrupt_numbers_14k/train.parquet \
 data.val_files=data/train/countdown_level5_50k/test.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=1024 \
 data.max_prompt_length=2048 \
 data.max_response_length=$MAX_LENGTH \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.0 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=1.0 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.000 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='countdown'\
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
 trainer.rejection_sample=True \
 trainer.push_to_hub=False 2>&1 | tee $CHECKPOINTS_DIR/$EXPERIMENT_NAME.log

 ## Push Saved Checkpoints to Huggingface
#  python3 push_to_hf/experiments.py --project_name countdown --run_name $EXPERIMENT_NAME