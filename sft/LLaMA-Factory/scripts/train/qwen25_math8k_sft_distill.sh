set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="math8k"

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25_math8k_sft_distill.yaml