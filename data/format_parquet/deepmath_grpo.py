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
from datasets import load_dataset, concatenate_datasets
from multiprocessing import Pool

def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        answer = example.pop('final_answer')
        difficulty = example.pop('difficulty')
        topic = example.pop('topic')

        data = {
            "data_source": 'deepmath',
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
    python data/format_parquet/deepmath_grpo.py --train_dir "./data/train/deepmath_level3-4" --test_dir "./data/test/deepmath_level6-7" --train_levels 3.0 3.5 4.0 4.5 --test_levels 6.0 6.5 7.0 7.5
    """
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    parser.add_argument('--train_levels', nargs='+', default=[5, 6, 7, 8, 9], type=float)
    parser.add_argument('--test_levels', nargs='+', default=[6,7], type=float)
    parser.add_argument('--train_dir', required=True, default='./data/train', help='Local directory to save processed datasets')
    parser.add_argument('--test_dir', required=True, default='./data/test', help='Local directory to save processed datasets')
    parser.add_argument('--tokenizer_name', default='Qwen/Qwen2.5-1.5B', help='Tokenizer name')
    parser.add_argument('--prompt_max_length', default=512, help='Max length')
    
    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    train_levels = args.train_levels
    test_levels = args.test_levels
    tokenizer_name = args.tokenizer_name
    prompt_max_length = args.prompt_max_length

    import pdb; pdb.set_trace()
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_dataset, test_dataset = load_dataset("aochongoliverli/DeepMath-103K")["train"], load_dataset("aochongoliverli/DeepMath-103K")["test"]
    train_dataset = train_dataset.filter(lambda x: x["difficulty"] in train_levels, num_proc=4)
    test_dataset = test_dataset.filter(lambda x: x["difficulty"] in test_levels, num_proc=4)

    # length filtering
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = train_dataset.filter(lambda x: remove_long_prompt(x["question"], tokenizer, prompt_max_length), num_proc=4)
    test_dataset = test_dataset.filter(lambda x: remove_long_prompt(x["question"], tokenizer, prompt_max_length), num_proc=4)
    
    remove_columns = ['r1_solution_1', 'r1_solution_2', 'r1_solution_3', 'subtopic']
    train_data = train_dataset.map(make_map_fn('train'), with_indices=True, num_proc=4, remove_columns=remove_columns)
    test_data = test_dataset.map(make_map_fn('test'), with_indices=True, num_proc=4, remove_columns=remove_columns)

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)    
    train_df.to_parquet(os.path.join(train_dir, 'train.parquet'))

    # Save test dataset
    print("test data size:", len(test_data))
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(test_dir, 'test.parquet'))