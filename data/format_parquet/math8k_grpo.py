import argparse
import os
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
from transformers import AutoTokenizer

import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, concatenate_datasets, load_from_disk

from multiprocessing import Pool
import argparse
def make_map_fn(split: str, source:str=None):
    def process_train_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        answer = example.pop('gt_answer')
        difficulty = example.pop('level')
        subject = example.pop('subject')

        data = {
            "data_source": source,
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
                'subject': subject
            }
        }
        return data
    
    def process_test_fn(example: Dict[str, Any], idx: int):
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

    return process_train_fn if split == 'train' else process_test_fn

if __name__ == '__main__':
    """
    python data/format_parquet/math8k_grpo.py \
        --aime_dir ../perturb-r/data/aime2425 \
        --amc_dir ../perturb-r/data/amc23 \
        --math500_dir ../perturb-r/data/math500 \
        --sft_name_or_path aochongoliverli/math8k-sft-QwQ-32B-reasoning-traces
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--aime_dir", type=str, default=None)
    parser.add_argument("--amc_dir", type=str, default=None)
    parser.add_argument("--math500_dir", type=str, default=None)
    parser.add_argument("--sft_name_or_path", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs("data/train/math8k", exist_ok=True)

    train_ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/train.parquet")["train"]
    aime_ds = load_from_disk(args.aime_dir)['test']
    amc_ds = load_from_disk(args.amc_dir)['test']
    math500_ds = load_from_disk(args.math500_dir)['test']
    test_ds = concatenate_datasets([aime_ds] * 4 + [amc_ds] * 4 + [math500_ds])

    train_ds = train_ds.map(make_map_fn("train", "math8k"), with_indices=True)
    test_ds = test_ds.map(make_map_fn("test", "test"), with_indices=True)
    
    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    train_df = pd.DataFrame(train_ds, columns=columns)
    test_df = pd.DataFrame(test_ds, columns=columns)

    if args.sft_name_or_path is not None:
        sft_data = load_dataset(args.sft_name_or_path, split="train")
        problems = set(sft_data["problem"])
        
        in_train = train_df.apply(lambda x: x["prompt"][0]["content"] in problems, axis=1)
        heldout = train_df.apply(lambda x: x["prompt"][0]["content"] not in problems, axis=1)

        heldout_df = train_df[heldout].reset_index(drop=True)
        train_df = train_df[in_train].reset_index(drop=True)

    train_df.to_parquet("data/train/math8k/train.parquet")
    test_df.to_parquet("data/train/math8k/test.parquet")
    heldout_df.to_parquet("data/train/math8k/heldout.parquet")