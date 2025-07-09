import argparse
import os
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm
from transformers import AutoTokenizer

import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, concatenate_datasets
from multiprocessing import Pool

def add_index_to_extra_info(row):
    extra_info = row['extra_info']
    extra_info['index'] = row['index']

    return extra_info

if __name__ == '__main__':
    """
    Example usage:
    python data/format_parquet/math8k_grpo.py --file_path "./data/train/math.8k/test.parquet"
    """
    parser = argparse.ArgumentParser(description='Process datasets for RL Training on DeepMath')
    parser.add_argument('--file_path', required=True)

    args = parser.parse_args()
    file_path = args.file_path

    df = pd.read_parquet(file_path).reset_index()
    df['extra_info'] = df.apply(add_index_to_extra_info, axis=1)
    df.drop(columns=['index'], inplace=True)

    df.to_parquet(file_path)