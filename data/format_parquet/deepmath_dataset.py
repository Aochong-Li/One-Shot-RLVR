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
from datasets import load_dataset, concatenate_datasets
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
    python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/"
    """
    import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    # parser.add_argument('--difficulty_levels', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--sample_size', type=int, default=20000)
    parser.add_argument('--local_dir', required=True, default='./data/train', help='Local directory to save processed datasets')
    
    args = parser.parse_args()
    local_dir = args.local_dir
    # difficulty_levels = args.difficulty_levels

    data_dir = os.path.join(local_dir, f'deepmath_level5-9')
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = load_dataset("aochongoliverli/DeepMath-103K-split")["train"]

    """
    We remove T/F Yes/No questions to make guessing the answer harder
    """
    train_dataset = train_dataset.filter(lambda x: x["final_answer"].lower() not in ["true", "false", "yes", "no"])
    
    train_dataset_5to6 = train_dataset.filter(lambda x: x["difficulty"] in [5, 6])
    train_dataset_7to9 = train_dataset.filter(lambda x: x["difficulty"] in [7, 8, 9])

    if args.sample_size is not None:
        subsample_size = args.sample_size // 2
        train_dataset_5to6 = train_dataset_5to6.shuffle(seed=42).select(range(subsample_size))
        train_dataset_7to9 = train_dataset_7to9.shuffle(seed=42).select(range(subsample_size))
        train_dataset = concatenate_datasets([train_dataset_5to6, train_dataset_7to9])
    
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