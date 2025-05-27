# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-system"
export CUDA_VISIBLE_DEVICES=0,1
MAX_TOKENS="4096"

MODEL_LIST=(
    "aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_600"
    "aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_1000"
    "aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_1400"
    "aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-4096-rollout-8-global_step_1800"
)
for MODEL in "${MODEL_LIST[@]}";do
    echo "======== Evaluating checkpoint at epoch: ${MODEL} ========"
    OUTPUT_DIR="./results/deepmath-hard-4096-rollout-8/${MODEL}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL $MAX_TOKENS $OUTPUT_DIR
done