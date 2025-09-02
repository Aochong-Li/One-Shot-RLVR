set -x

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="inception"

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25_gptoss_sft.yaml