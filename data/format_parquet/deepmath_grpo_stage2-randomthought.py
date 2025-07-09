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

if __name__ == '__main__':
    """
    Example usage:
    """
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    parser.add_argument('--stage2_train_filepath', required=True)
    parser.add_argument('--distractor_dataset_path', required=True)
    parser.add_argument('--train_dir', required=True, default='./data/train', help='Local directory to save processed datasets')
        
    args = parser.parse_args()
    train_filepath = args.stage2_train_filepath
    distractor_dataset_path = args.distractor_dataset_path
    train_dir = args.train_dir

    os.makedirs(train_dir, exist_ok=True)

    train_df = pd.read_parquet(train_filepath)
    distractor_dataset = pd.read_pickle(distractor_dataset_path)

    



    train_df.to_parquet(os.path.join(train_dir, 'train.parquet'))
