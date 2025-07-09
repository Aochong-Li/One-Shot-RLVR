"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, concatenate_datasets
from multiprocessing import Pool

def make_map_fn(split: str, source:str=None):
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        answer = example.pop('final_answer')
        difficulty = example.pop('difficulty')
        topic = example.pop('topic')

        data = {
            "data_source": source if source else f'deepmath-level{difficulty}',
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'difficulty': difficulty,
                'topic': topic
            }
        }
        return data
    return process_fn

def post_cutoff_reward(example, *, last_epoch_step):
    """
    Mean reward for timesteps > cutoff_step.
    Returns a dict so 🤗 Datasets adds it as a new column.
    """
    rewards = np.asarray(example["reward"])
    steps   = np.asarray(example["global_step"])
    mask    = steps > last_epoch_step
    return {
        f"avg_reward_after_step{last_epoch_step}": rewards[mask].mean() if mask.any() else np.nan
    }

def remove_tokenization(example, template_prefix, template_suffix):
    return {
        "question": example['question'].replace(template_prefix, "").replace(template_suffix, "")
    }

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/deepmath_grpo_stage2.py \
        --train_filepath ./data/train/deepmath_level3-4/train.parquet \
        --stage1_rollout_hf_dataset_path aochongoliverli/R1-Distill-Qwen-1.5B-DeepMath-stage1-grpo-level3-4-5epochs-4rollouts-8192max-length-rollouts \
        --last_epoch_step 285 \
        --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --train_dir ./data/train/deepmath_level3-4-stage2-post-step285-reward-lt-1.0
    """
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    parser.add_argument('--train_filepath', required=True)
    parser.add_argument('--stage1_rollout_hf_dataset_path', required=True)
    parser.add_argument('--last_epoch_step', type=int, required=True)
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--train_dir', required=True, default='./data/train', help='Local directory to save processed datasets')
        
    args = parser.parse_args()
    train_filepath = args.train_filepath
    stage1_rollout_hf_dataset_path = args.stage1_rollout_hf_dataset_path
    last_epoch_step = args.last_epoch_step
    train_dir = args.train_dir
    tokenizer_name = args.tokenizer_name

    os.makedirs(train_dir, exist_ok=True)

    train_df = pd.read_parquet(train_filepath)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    stage1_rollout_dataset = load_dataset(stage1_rollout_hf_dataset_path)['train']
    stage1_rollout_dataset = stage1_rollout_dataset.map(post_cutoff_reward, fn_kwargs={"last_epoch_step": last_epoch_step}, num_proc=4)

    subset_dataset = stage1_rollout_dataset.filter(lambda x: x[f"avg_reward_after_step{last_epoch_step}"] < 1.0)
    template_prefix, template_suffix = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': "HANDLE"}],
        add_generation_prompt=True,
        tokenize=False
    ).split("HANDLE")
    subset_dataset = subset_dataset.map(remove_tokenization, fn_kwargs={"template_prefix": template_prefix, "template_suffix": template_suffix}, num_proc=4)
    subset_questions = set(subset_dataset["question"])
    zero_success_questions = set(subset_dataset.filter(lambda x: x[f"avg_reward_after_step{last_epoch_step}"] == 0.0)["question"])

    import pdb; pdb.set_trace()
    mask = train_df.apply(lambda x: x['prompt'][0]['content'] in subset_questions, axis=1)
    train_df = train_df[mask]
    train_df['zero_success'] = train_df['prompt'].apply(lambda x: x[0]['content'] in zero_success_questions)
    print("train data size:", len(train_df))

    train_df.to_parquet(os.path.join(train_dir, 'train.parquet'))
