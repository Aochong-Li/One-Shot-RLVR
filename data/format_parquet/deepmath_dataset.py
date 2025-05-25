"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset

def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int, data_source: str) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        # instruction = "Let's think step by step and output the final answer within \\boxed{}."
        # question = f"{question} {instruction}"
        answer = example.pop('answer')

        data = {
            "data_source": data_source,
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
                'index': idx
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/deepmath_4096"
    """
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', required=True, help='Local directory to save processed datasets')
    args = parser.parse_args()
    local_dir = args.local_dir
    
    # Make local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_dataset = load_dataset("aochongoliverli/deepmath-4096-hard-rl")["train"]
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')

    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx, "deepmath-4096-hard")
        if processed_example is not None:
            train_data.append(processed_example)


    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))