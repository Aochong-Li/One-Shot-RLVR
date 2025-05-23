# How to filter OpenR1 220K to 15K?
# 1. only keep questions that R1 answer correctly all the time
# 2. remove easy questions, success rate >50% by some similar model 
import datasets
from datasets import load_dataset, DatasetDict, Dataset
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
def difficulty_filtering_openr1_dataset(output_path: str,
                                         max_success_rate: float = 0.75,
                                         min_success_rate: float = 0.01,
                                         coldstart_size: int = 1000,
                                         max_length: int = 16384,
                                         tokenizer_name: str = "Qwen/Qwen2.5-1.5B"
                                         ):
    bigmath = load_dataset("SynthLabsAI/Big-Math-RL-Verified")
    bigmath = bigmath['train'].filter(lambda x: x['llama8b_solve_rate'] is not None and x['llama8b_solve_rate'] <= max_success_rate and x['llama8b_solve_rate'] >= min_success_rate)
    bigmath = bigmath.map(lambda x: {'problem': x['problem'].strip('\n')})

    # Load the OpenR1 220K dataset
    openthought_114k = load_dataset("open-r1/OpenThoughts-114k-math")['train']
    openthought_114k = openthought_114k.map(lambda x: {'problem': x['problem'].strip('\n')})
    problem_set = set(openthought_114k['problem'])

    
    bigmath_filtered = bigmath.filter(lambda x: x['problem'] in problem_set)
    problem_set = set(bigmath_filtered['problem'])
    openthought_114k = openthought_114k.filter(lambda x: x['problem'] in problem_set)

    # merge two datasets 
    bigmath_df = bigmath_filtered.remove_columns(['source', 'domain']).to_pandas()
    openthought_df = openthought_114k.to_pandas()

    df = openthought_df.merge(bigmath_df, on='problem')

    def check_answer_consistency(row):
        if row['answer'].strip() in row['solution'].strip():
            return True
        else:
            return False

    consistent_answer = df.apply(check_answer_consistency, axis=1)
    # Convert to Dataset without the index
    df_consistent = df[consistent_answer].reset_index(drop=True)

    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def tokenized_len (example, tokenizer = tokenizer):
        question = example['conversations'][0]['value']
        reasoning = example['conversations'][-1]['value']
        prompt = question + "\n" + reasoning

        return len(tokenizer.encode(prompt))
    import pdb; pdb.set_trace()
    df_consistent['tokenized_len'] = df_consistent.apply(tokenized_len, axis=1)
    df_consistent = df_consistent[df_consistent['tokenized_len'] <= max_length].drop(columns=['tokenized_len'])
    
    train_df, coldstart_df = train_test_split(df_consistent, test_size=coldstart_size, random_state=42)
    dataset = DatasetDict({'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
                           'coldstart': Dataset.from_pandas(coldstart_df.reset_index(drop=True))})

    # Save to disk without the index parameter
    dataset.save_to_disk(output_path)

if __name__ == "__main__":
    output_path = '/share/goyal/lio/reasoning/data/openthought-hard/dataset'
    difficulty_filtering_openr1_dataset(output_path)
