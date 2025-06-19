"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm

import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset
from multiprocessing import Pool

def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int, data_source: str) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        answer = example.pop('final_answer')
        difficulty = example.pop('difficulty')
        topic = example.pop('topic')

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
                'index': idx,
                'difficulty': difficulty,
                'topic': topic
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/" --difficulty_levels 5 6
    python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/" --difficulty_levels 7 8 9
    """
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    parser.add_argument('--difficulty_levels', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--sample_size', type=int, default=20000)
    parser.add_argument('--local_dir', required=True, default='./data/train', help='Local directory to save processed datasets')
    
    args = parser.parse_args()
    local_dir = args.local_dir
    difficulty_levels = args.difficulty_levels

    data_dir = os.path.join(local_dir, f'deepmath_level{min(difficulty_levels)}-{max(difficulty_levels)}')
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = load_dataset("aochongoliverli/DeepMath-103K-split")["train"].filter(lambda x: x['difficulty'] in difficulty_levels)
    """
    We remove T/F Yes/No questions to make guessing the answer harder
    """
    train_dataset = train_dataset.filter(lambda x: x["final_answer"].lower() not in ["true", "false", "yes", "no"])

    if args.sample_size is not None:
        train_dataset = train_dataset.shuffle(seed=42).select(range(args.sample_size))
    process_fn = make_map_fn('train')

    def process_dataset(args):
        idx, example = args
        return process_fn(example, idx, "deepmath")

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(process_dataset, enumerate(train_dataset)), total=len(train_dataset)))
    train_data = [data for data in results if data is not None]

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)    
    train_df.to_parquet(os.path.join(data_dir, 'train.parquet'))