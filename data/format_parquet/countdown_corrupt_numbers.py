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
                'level': str(level)
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
    DATA_FILEPATH="/mnt/home/al2644/research/projects/perturb-r/results/countdown_train_stage2_level5_100K/corrupt_numbers/Qwen2.5-3B-countdown-level4-5-stage1_rl.pickle"
    python data/format_parquet/countdown_corrupt_numbers.py --local_dir "./data/train/countdown_level5_corrupt_numbers_7k" --corrupt_datafile $DATA_FILEPATH --tokenizer_name "aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch" --max_prompt_length 2048
    python data/format_parquet/countdown_corrupt_numbers.py --local_dir "./data/train/countdown_level5_corrupt_numbers_14k" --corrupt_datafile $DATA_FILEPATH --tokenizer_name "aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch" --max_prompt_length 2048
    """
    parser = argparse.ArgumentParser(description='Process datasets for Countdown training')
    parser.add_argument('--local_dir', required=True, help='Local directory to save processed datasets')
    parser.add_argument('--corrupt_datafile', required=True, help='Corrupt data file')
    parser.add_argument('--tokenizer_name', required=True, help='Tokenize name')
    parser.add_argument('--max_prompt_length', type=int, required=True, default=2048, help='Max prompt length')

    args = parser.parse_args()
    local_dir = args.local_dir
    corrupt_datafile = args.corrupt_datafile
    tokenizer_name = args.tokenizer_name

    # Load corrupt data
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    corrupt_df = pd.read_pickle(corrupt_datafile)

    corrupt_df = corrupt_df[corrupt_df["still_correct"] == 0.][["nums", "target", "level", "prompt"]]
    corrupt_df["prompt_token_length"] = corrupt_df["prompt"].apply(lambda x: len(tokenizer.encode(x)))
    corrupt_df = corrupt_df[corrupt_df["prompt_token_length"] <= args.max_prompt_length].reset_index(drop=True)
    corrupt_dataset = Dataset.from_pandas(corrupt_df)

    template_prefix, template_suffix = tokenizer.apply_chat_template([{"role":"user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True).split("HANDLE")

    os.makedirs(local_dir, exist_ok=True)
    
    corrupt_data = corrupt_dataset.map(
        corrupt_make_map_fn('train'),
        fn_kwargs={
            "template_prefix": template_prefix,
            "template_suffix": template_suffix
        },
        with_indices=True,
        num_proc=10,
        remove_columns=["prompt_token_length"]
    )
    original_data = corrupt_dataset.map(
        make_map_fn('train'),
        with_indices=True,
        num_proc=10,
        remove_columns=["prompt_token_length"]
    )
    import pdb; pdb.set_trace()

    df = pd.DataFrame(concatenate_datasets([original_data, corrupt_data]))
    
    print("train data size:", len(df))

    df.to_parquet(os.path.join(local_dir, 'train.parquet'))