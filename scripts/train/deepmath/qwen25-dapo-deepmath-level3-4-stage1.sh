#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR="./outputs"
STAGE0_STEP=1400
export BASE_MODEL="aochongoliverli/Qwen2.5-1.5B-DeepMath-level1-4-40k-all_rollouts-sft-stage0-step-${STAGE0_STEP}" #TODO: change this

N_GPUS=8
ROLLOUT_N=4
EXPECTED_MAX_LENGTH=4096
VALIDATION_MAX_LENGTH=16384
TENSOR_MODEL_PARALLEL_SIZE=1
TOTAL_EPOCHS=10 # Set it large enough and we can stop it early
SAVE_STEPS=50
EVAL_STEPS=50

OVERLONG_BUFFER_LEN=512
OVERLONG_BUFFER_PENALTY_FACTOR=0.2
MAX_LENGTH=$((EXPECTED_MAX_LENGTH + OVERLONG_BUFFER_LEN))

EXPERIMENT_NAME="Qwen2.5-1.5B-DeepMath-sft-step${STAGE0_STEP}-stage1-dapo-level3-4-rollout-${ROLLOUT_N}-max-len-${MAX_LENGTH}"

python3 -m verl.trainer.main_dapo \
 algorithm.adv_estimator=dapo \
 data.train_files=./data/train/deepmath_level3-4/train.parquet \
 data.val_files=./data/test/deepmath_level6-8/test.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=300 \
 data.max_prompt_length=512 \
 data.max_response_length=${MAX_LENGTH} \
 reward_model.reward_manager='dapo' \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
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
 actor_rollout_ref.rollout.val_response_length=$VALIDATION_MAX_LENGTH \
 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.0 \
 reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
 reward_model.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='deepmath'\
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