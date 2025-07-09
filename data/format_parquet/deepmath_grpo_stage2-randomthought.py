"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
import pandas as pd


def add_random_thought(rows, distractor_dataset):
    prompt = rows['prompt']
    user_input = prompt[0]['content']

    if user_input not in list(distractor_dataset['problem']):
        return [
            {"role": "user", "content": user_input}
        ]
    
    random_thought = distractor_dataset[distractor_dataset['problem'] == user_input]['random_thought'].values[0]

    return [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": random_thought}
    ]

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/deepmath_grpo_stage2-randomthought.py \
        --stage2_train_filepath data/train/deepmath_level3-4-stage2-post-step285-reward-lt-1.0/train.parquet \
        --distractor_dataset_path "/mnt/home/al2644/research/projects/perturb-r/results/deepmath_generate_data/inject_thoughts/Qwen2.5-1.5B-DeepMath-stage1-grpo-level3-4-5epochs-4rollouts-8192max-length-5epochs.pickle" \
        --train_dir data/train/deepmath_level3-4-stage2-post-step285-reward-lt-1.0-randomthought
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
    distractor_dataset['random_thought'] = distractor_dataset['random_thought'].apply(lambda x: "<think>\n" + x)
    train_df['prompt'] = train_df.apply(lambda x: add_random_thought(x, distractor_dataset), axis=1)

    train_df.to_parquet(os.path.join(train_dir, 'train.parquet'))
