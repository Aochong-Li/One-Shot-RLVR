set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="countdown"

# conda activate llama-factory
cd /mnt/home/al2644/research/projects/rlvr/sft/LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25_3b_countdown_stage2_ablation_sft.yaml 2>&1 | tee logs/qwen25_3b_countdown_stage2_ablation_sft.log

## Push checkpoints to huggingface
# python3 push_to_hf/experiments.py --project_name countdown --run_name Qwen2.5-3B-sft-distill-countdown-level5-corrupt-numbers-stage2-ablation
