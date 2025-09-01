import argparse
import os
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
from transformers import AutoTokenizer

import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, DatasetDict

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

def convert_to_hf_dataset(example: Dict[str, Any]):
    source = example.pop('data_source')
    prompt = example.pop('prompt')
    reward_model = example.pop('reward_model')
    extra_info = example.pop('extra_info')
    
    problem = prompt[0]['content']
    solution = reward_model['ground_truth']
    level = extra_info['difficulty']
    subject = extra_info['subject']

    return {
        "problem": problem,
        "solution": solution,
        "level": level,
        "subject": subject,
        "source": source
    }

if __name__ == '__main__':
    """
    # RL on all data
    python data/format_parquet/math8k_grpo.py \
        --math500_dir ../perturb-r/data/math500 \
        --sft_name_or_path aochongoliverli/math8k-sft-QwQ-32B-16k-reasoning-traces
    
    # RL excluding coldstart data
    python data/format_parquet/math8k_grpo.py \
        --coldstart_name_or_path aochongoliverli/math8k-coldstart-QwQ-32B-reasoning-traces
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--coldstart_name_or_path", type=str, default=None)
    parser.add_argument("--aime_dir", type=str, default=None)
    parser.add_argument("--amc_dir", type=str, default=None)
    parser.add_argument("--math500_dir", type=str, default=None)
    parser.add_argument("--sft_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, choices=["simplelr_qwen_gsm8k_level1", "simplelr_qwen_level1to4","simplelr_qwen_level3to5"])
    parser.add_argument("--dataset_short_name", type=str, required=True, default="math8k")
    parser.add_argument("--push_to_hf", action="store_true")
    args = parser.parse_args()
    
    import pdb; pdb.set_trace()
    if args.coldstart_name_or_path is not None and os.path.exists(f"data/train/{args.dataset_short_name}/train.parquet"):
        df = pd.read_parquet(f"data/train/{args.dataset_short_name}/train.parquet")
        coldstart_data = load_dataset(args.coldstart_name_or_path, split="train")
        coldstart_problems = set(coldstart_data["problem"])

        in_train = df.apply(lambda x: x["prompt"][0]["content"] not in coldstart_problems, axis=1)
        train_df = df[in_train].reset_index(drop=True)
        train_df.to_parquet(f"data/train/{args.dataset_short_name}/coldstart_rl_train.parquet")
    
    else:
        os.makedirs(f"data/train/{args.dataset_short_name}", exist_ok=True)

        train_ds = load_dataset("parquet", data_files=f"https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/{args.dataset_name}/train.parquet")["train"]
        test_ds = []
        if args.aime_dir is not None:
            test_ds.append(4 * load_from_disk(args.aime_dir)['test'])
        if args.amc_dir is not None:
            test_ds.append(4 * load_from_disk(args.amc_dir)['test'])
        if args.math500_dir is not None:
            test_ds.append(load_from_disk(args.math500_dir)['test'])
        test_ds = concatenate_datasets(test_ds)

        train_ds = train_ds.map(make_map_fn("train", args.dataset_short_name), with_indices=True)
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
            
            heldout_df.to_parquet(f"data/train/{args.dataset_short_name}/heldout.parquet")

        train_df.to_parquet(f"data/train/{args.dataset_short_name}/train.parquet")
        test_df.to_parquet(f"data/train/{args.dataset_short_name}/test.parquet")

        if args.push_to_hf:
            dataset =  Dataset.from_pandas(train_df)
            dataset = dataset.map(convert_to_hf_dataset).remove_columns(["ability"])

            repo_id = f"aochongoliverli/{args.dataset_short_name}"
            dataset_dict = DatasetDict({
                "test": dataset
            })
            
            dataset_dict.push_to_hub(repo_id, private=False)