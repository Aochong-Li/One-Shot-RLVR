set -x

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="math8k"

# 3B Full Fine-Tuning
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_qwq.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_am.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_qwen3.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_3b_math8k_qwq_limo.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_limo/qwen25_3b_math8k_limo.yaml

# 1.5B Full Fine-Tuning
# FORCE_TORCHRUN=1 llamafactory-clitrain examples/train_distill_math8k/qwen25_1.5b_math8k_am.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_1.5b_math8k_qwq.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_1.5b_math8k_qwen3.yaml

# 7B LoRA-Tuning
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_7b_math8k_qwq.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_7b_math8k_am.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_distill_math8k/qwen25_7b_math8k_qwen3.yaml