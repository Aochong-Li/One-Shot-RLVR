set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="llamafactory-sft"

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25_math_full_self_distill.yaml

## Push checkpoints to huggingface
# python3 push_to_hf/experiments.py --project_name llamafactory-sft --run_name Qwen2.5-Math-1.5B-DeepMath-Hard-GRPO-1800-steps-SFT
