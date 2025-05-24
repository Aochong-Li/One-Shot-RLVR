<div align="center">

# Reinforcement Learning for Reasoning in Large Language Models with One Training Example


[Oliver Li](https://github.com/Aochong-Li)

</div>

<!-- ## Updates
* 17/05/2025: We release our [checkpoints](https://huggingface.co/collections/ypwang61/one-shot-rlvr-6827f72c3359b2ffe75fc1a8) and [dataset](https://huggingface.co/datasets/ypwang61/one_shot_rlvr) in huggingface.
* 30/04/2025: 🎉 We release our [paper](https://arxiv.org/abs/2504.20571), [code](https://github.com/ypwang61/One-Shot-RLVR), and [wandb records](https://wandb.ai/yipingwanguw/verl_few_shot?nw=nwuseryipingwang22). See the summarization of our work at [X(twitter)](https://x.com/ypwang61/status/1917596101953348000). -->

# Instruction 

## Change to the directory
```bash
cd rlvr
git pull
git checkout grpo
conda activate zero
```

## Upload model weights
```bash
 python push_to_hf/experiments.py --experiment_name Qwen2.5-Math-1.5B-dsr_sub
 ```

 ## Launch GRPO jobs on DeepMath
 ```bash
 bash scripts/train/deepmath/training_1.5b_deepmath-8a100.sh
 ```