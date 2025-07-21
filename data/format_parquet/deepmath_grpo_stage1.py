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
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, concatenate_datasets, load_from_disk
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
                'difficulty': str(difficulty),
                'topic': topic
            }
        }
        return data
    return process_fn

def process_aime_fn(example: Dict[str, Any], idx: int):
    problem = example.pop('problem')
    solution = example.pop('solution')
    source = example.pop('source')
    
    data = {
        "data_source": source,
        "prompt": [{
            "role": "user",
            "content": problem
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': 'test',
            'index': idx,
            'difficulty': source,
            'topic': 'aime'
        }
    }
    return data

def remove_long_prompt(prompt, tokenizer, max_length):
    tokenized = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        add_generation_prompt=True,
        tokenize=False
        )
    if len(tokenized) > max_length:
        return False
    
    return True

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/deepmath_grpo_stage1.py \
        --train_dir "./data/train/deepmath_level5" \
        --test_dir "./data/test/deepmath_level5-9" \
        --train_levels 5.0 5.5 \
        --test_levels 5.0 6.0 7.0 8.0 9.0 \
        --test_size_per_level 50 \
        --tokenizer_name Qwen/Qwen2.5-1.5B \
        --prompt_max_length 512
    """
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    parser.add_argument('--train_levels', nargs='+', default=[5, 6, 7, 8, 9], type=float)
    parser.add_argument('--test_levels', nargs='+', default=[6,7], type=float)
    parser.add_argument('--aime_dir', default=None)
    parser.add_argument('--train_dir', required=True, default='./data/train', help='Local directory to save processed datasets')
    parser.add_argument('--test_dir', required=True, default='./data/test', help='Local directory to save processed datasets')
    parser.add_argument('--test_size_per_level', type=int, default=200, help='Test size per level')
    parser.add_argument('--tokenizer_name', default='Qwen/Qwen2.5-1.5B', help='Tokenizer name')
    parser.add_argument('--prompt_max_length', type=int, default=512, help='Max length')
    import pdb; pdb.set_trace()
    
    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    train_levels = args.train_levels
    test_levels = args.test_levels
    test_size_per_level = args.test_size_per_level
    tokenizer_name = args.tokenizer_name
    prompt_max_length = args.prompt_max_length
    aime_dir = args.aime_dir

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_dataset, test_dataset = load_dataset("aochongoliverli/DeepMath-103K")["train"], load_dataset("aochongoliverli/DeepMath-103K")["test"]
    train_dataset = train_dataset.filter(lambda x: x["difficulty"] in train_levels, num_proc=4)
    test_dataset = test_dataset.filter(lambda x: x["difficulty"] in test_levels, num_proc=4)

    # length filtering
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = train_dataset.filter(lambda x: remove_long_prompt(x["question"], tokenizer, prompt_max_length), num_proc=4)
    train_dataset = train_dataset.filter(lambda x: x["final_answer"].lower() not in ["true", "false", "yes", "no", 'a', 'b', 'c', 'd', 'e'])
    test_dataset = test_dataset.filter(lambda x: remove_long_prompt(x["question"], tokenizer, prompt_max_length), num_proc=4)
    test_dataset = test_dataset.filter(lambda x: x["final_answer"].lower() not in ["true", "false", "yes", "no", 'a', 'b', 'c', 'd', 'e'])

    remove_columns = ['r1_solution_1', 'r1_solution_2', 'r1_solution_3', 'subtopic']
    train_data = train_dataset.map(make_map_fn('train', source='deepmath'), with_indices=True,  num_proc=4, remove_columns=remove_columns)
    test_data = test_dataset.map(make_map_fn('test'), with_indices=True, num_proc=4, remove_columns=remove_columns)

    if aime_dir:
        aime_dataset = load_from_disk(aime_dir)['test']
        aime_dataset = aime_dataset.map(process_aime_fn, with_indices=True, num_proc=4)
        test_data = concatenate_datasets([test_data, aime_dataset])

    # Save training dataset
    train_df = pd.DataFrame(train_data)    
    print("train data size:", len(train_df))
    train_df.to_parquet(os.path.join(train_dir, 'train.parquet'))

    # Save test dataset
    test_df = pd.DataFrame(test_data)
    test_df = test_df.groupby("data_source").apply(lambda x: x.sample(n=test_size_per_level, random_state=42)).reset_index(drop=True)

    print("test data size:", len(test_df))
    test_df.to_parquet(os.path.join(test_dir, 'test.parquet'))