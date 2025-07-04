import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        numbers = example.pop('nums')
        target = example.pop('target')
        level = example.pop('level')

        question = f"Please reason and answer the following question. Using the numbers {numbers}, create an equation that equals {target}. You can only use basic arithmetic operations (+, -, *, /) in the expression and each number should be used exactly once. You should report the answer equation expression in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."

        data = {
            "data_source": "countdown",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    'numbers': numbers,
                    'target': target
                }
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'level': level
            }
        }
        return data
    return process_fn

def corrupt_make_map_fn(split: str):
    def process_fn(example, idx, template_prefix: str, template_suffix: str) -> Optional[Dict[str, Any]]:
        prompt = example.pop('prompt')
        user_content, assistant_content = prompt.split(template_suffix)
        user_content = user_content.replace(template_prefix, '')
        numbers, target, level = example.pop('nums'), example.pop('target'), example.pop('level')

        data = {
            "data_source": "countdown-corrupt",
            "prompt": [{
                    "role": "user",
                    "content": user_content
                    },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    'numbers': numbers,
                    'target': target
                }
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'level': str(level)
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/countdown_random_thought.py --local_dir "./data/train/countdown_level5_random_thought_only_15k" --test_levels 6 7 --used_size 15000 --test_size_per_level 150 --tokenizer_name "aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch" --corrupt_datafile "/mnt/home/al2644/research/projects/perturb-r/results/countdown_stage2_generate/inject_thoughts/Qwen2.5-3B-countdown-level4-5-stage1_rl.pickle"
    """
    parser = argparse.ArgumentParser(description='Process datasets for Countdown training')
    parser.add_argument('--local_dir', required=True, help='Local directory to save processed datasets')
    parser.add_argument('--corrupt_datafile', required=True, help='Corrupt data file')
    parser.add_argument('--tokenizer_name', required=True, help='Tokenize name')
    parser.add_argument('--train_levels', nargs='+', default=[], help='Levels to process')
    parser.add_argument('--test_levels', nargs='+', default=[], help='Levels to process')
    parser.add_argument('--used_size', type=int, default=5000)
    parser.add_argument('--train_size', type=int, default=20000)
    parser.add_argument('--test_size_per_level', type=int, default=100)

    args = parser.parse_args()
    local_dir = args.local_dir
    corrupt_datafile = args.corrupt_datafile
    tokenizer_name = args.tokenizer_name
    train_levels = args.train_levels
    test_levels = args.test_levels
    used_size = args.used_size
    train_size = args.train_size
    test_size_per_level = args.test_size_per_level

    train_size_per_level = train_size // len(train_levels) if train_levels else 0

    # Load corrupt data
    corrupt_df = pd.read_pickle(corrupt_datafile)
    corrupt_df = corrupt_df[corrupt_df["still_correct"] == 0.][["nums", "target", "level", "prompt"]].reset_index(drop=True)
    corrupt_dataset = Dataset.from_pandas(corrupt_df)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    template_prefix, template_suffix = tokenizer.apply_chat_template([{"role":"user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True).split("HANDLE")

    # Load original data

    # Make local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    train_dataset, test_dataset = [], []
    for level in train_levels:
        dataset = load_dataset(f"aochongoliverli/countdown_level_{level}")
        dataset = dataset['train'].select(range(used_size, used_size + train_size_per_level))
        dataset = dataset.add_column('level', [level] * len(dataset))
        train_dataset.append(dataset)

    for level in test_levels:
        dataset = load_dataset(f"aochongoliverli/countdown_level_{level}")
        dataset = dataset['test'].select(range(test_size_per_level))
        dataset = dataset.add_column('level', [level] * len(dataset))
        test_dataset.append(dataset)

    train_dataset = concatenate_datasets(train_dataset) if train_levels else None
    test_dataset = concatenate_datasets(test_dataset)

    remove_columns = ['solution']
    corrupt_data = corrupt_dataset.map(
        corrupt_make_map_fn('train'),
        fn_kwargs={
            "template_prefix": template_prefix,
            "template_suffix": template_suffix
        },
        with_indices=True,
        num_proc=10
    )

    train_data = train_dataset.map(make_map_fn('train'), with_indices=True, num_proc=10).remove_columns(remove_columns) if train_levels else None
    test_data = test_dataset.map(make_map_fn('test'), with_indices=True, num_proc=10).remove_columns(remove_columns)

    train_df = pd.DataFrame(train_data)
    corrupt_df = pd.DataFrame(corrupt_data)
    
    train_df = pd.concat([train_df, corrupt_df], ignore_index=True) if train_levels else corrupt_df
    test_df = pd.DataFrame(test_data)

        # Save training dataset
    print("train data size:", len(train_df))
    print("test data size:", len(test_df))

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))