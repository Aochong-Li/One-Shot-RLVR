# rm -rf sh/eval_checkpoint_yiping.sh; vim sh/eval_checkpoint_yiping.sh
PROMPT_TYPE="qwen25-math-think"
export CUDA_VISIBLE_DEVICES=0,1
MAX_TOKENS="4096"

MODEL_LIST=(
    "aochongoliverli/Qwen2.5-Math-1.5B-dsr_sub-global_step_1350"
    "aochongoliverli/Qwen2.5-Math-1.5B-dsr_sub-global_step_1000"
    "aochongoliverli/Qwen2.5-Math-1.5B-dsr_sub-global_step_800"
    "aochongoliverli/Qwen2.5-Math-1.5B-dsr_sub-global_step_500"
    "aochongoliverli/Qwen2.5-Math-1.5B-dsr_sub-global_step_200"
    "aochongoliverli/Qwen2.5-Math-1.5B-dsr_sub-global_step_0"
)
for MODEL in "${MODEL_LIST[@]}";do
    echo "======== Evaluating checkpoint at epoch: ${MODEL} ========"
    OUTPUT_DIR="/share/goyal/lio/reasoning/eval/benchmarks/market/${MODEL}"

    mkdir -p $OUTPUT_DIR

    bash sh/eval_all_math.sh $PROMPT_TYPE $MODEL $MAX_TOKENS $OUTPUT_DIR
done