#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
export BASE_MODEL="aochongoliverli/Qwen2.5-3B-math8k-sft-distill-20epochs-5e-5lr-step150"

N_GPUS=2
ROLLOUT_N=8
EXPECTED_MAX_LENGTH=8192
TENSOR_MODEL_PARALLEL_SIZE=1
TOTAL_EPOCHS=10
SAVE_STEPS=20
EVAL_STEPS=10

OVERLONG_BUFFER_LEN=0
OVERLONG_BUFFER_PENALTY_FACTOR=0.0
MAX_LENGTH=$((EXPECTED_MAX_LENGTH + OVERLONG_BUFFER_LEN))

EXPERIMENT_NAME="Qwen2.5-3B-math8k-sft-distill-150steps-dapo-${TOTAL_EPOCHS}epochs-${ROLLOUT_N}rollouts-${MAX_LENGTH}max-len"

python3 -m verl.trainer.main_dapo \
 algorithm.adv_estimator=dapo \
 data.train_files=data/train/math8k/train.parquet \
 data.val_files=data/train/math8k/test.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=1024 \
 data.max_prompt_length=256 \
 data.max_response_length=$MAX_LENGTH \
 reward_model.reward_manager='dapo' \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.kl_loss_coef=0.0 \
 actor_rollout_ref.actor.entropy_coeff=0.0 \
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
 actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.0 \
 reward_model.overlong_buffer.enable=False \
 reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
 reward_model.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='math8k'\
 trainer.username='aochongoliverli' \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 trainer.resume_mode='auto' \
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