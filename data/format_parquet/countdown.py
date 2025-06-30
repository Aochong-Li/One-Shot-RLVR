import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from datasets import load_dataset, concatenate_datasets

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

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/countdown.py --local_dir "./data/train/countdown" --train_levels 4 5 --test_levels 6 7 --sft_size 5000 --train_size 20000 --test_size_per_level 150
    """
    parser = argparse.ArgumentParser(description='Process datasets for Countdown training')
    parser.add_argument('--local_dir', required=True, help='Local directory to save processed datasets')
    parser.add_argument('--train_levels', nargs='+', required=True, help='Levels to process')
    parser.add_argument('--test_levels', nargs='+', required=True, help='Levels to process')
    parser.add_argument('--sft_size', type=int, required=True, default=5000)
    parser.add_argument('--train_size', type=int, required=True, default=20000)
    parser.add_argument('--test_size_per_level', type=int, required=True, default=100)

    args = parser.parse_args()
    local_dir = args.local_dir
    train_levels = args.train_levels
    test_levels = args.test_levels
    sft_size = args.sft_size
    train_size = args.train_size
    test_size_per_level = args.test_size_per_level

    train_size_per_level = train_size // len(train_levels)

    # Make local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    train_dataset, test_dataset = [], []
    for level in train_levels:
        dataset = load_dataset(f"aochongoliverli/countdown_level_{level}")
        dataset = dataset['train'].select(range(args.sft_size, args.sft_size + train_size_per_level))
        dataset = dataset.add_column('level', [level] * len(dataset))
        train_dataset.append(dataset)

    for level in test_levels:
        dataset = load_dataset(f"aochongoliverli/countdown_level_{level}")
        dataset = dataset['test'].select(range(test_size_per_level))
        dataset = dataset.add_column('level', [level] * len(dataset))
        test_dataset.append(dataset)

    train_dataset = concatenate_datasets(train_dataset)
    test_dataset = concatenate_datasets(test_dataset)

    remove_columns = ['solution']
    train_data = train_dataset.map(make_map_fn('train'), with_indices=True, num_proc=10).remove_columns(remove_columns)
    test_data = test_dataset.map(make_map_fn('test'), with_indices=True, num_proc=10).remove_columns(remove_columns)

    # Save training dataset
    print("train data size:", len(train_data))
    print("test data size:", len(test_data))

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))