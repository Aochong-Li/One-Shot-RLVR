set -x

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="llamafactory-sft"

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25_math_full_sft.yaml

## Push checkpoints to huggingface
python3 push_to_hf/experiments.py --project_name llamafactory-sft --run_name Qwen2.5-Math-1.5B-DeepMath-Hard-SFT
