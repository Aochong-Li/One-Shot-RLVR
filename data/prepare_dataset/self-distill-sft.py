import datasets
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, load_from_disk
import os
import sys
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def process_distill_sft_dataset(dataset_name: str,
                                max_responses: int = 8,
                                model_name: str = "Qwen/Qwen2.5-Math-1.5B",
                                final_dataset_name: str = "Qwen2.5-Math-1.5B-deepmath-hard-1800-steps-4096-self-distill-sft",
                                username: str = "aochongoliverli"
                                ):
    dataset = load_dataset(dataset_name)['train']
    dataset = dataset.filter(lambda x: x['correct_responses'] != [])

    def fill_responses(example):
        while len(example['correct_responses']) < max_responses:
            example['correct_responses'].append(np.random.choice(example['correct_responses']))
        return example
    
    dataset = dataset.map(fill_responses)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def format_dataset(batch, tokenizer = tokenizer):
        conversations, generated_token_counts = [], []
        n = len(batch['question'])
        for i in tqdm(range(n), desc="Formatting dataset", leave=True):
            question = batch['question'][i].split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip("\n")
            correct_responses = batch['correct_responses'][i]
            for response in correct_responses:
                conversation = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response}
                    ]
                conversations.append(conversation)
                generated_token_counts.append(len(tokenizer.apply_chat_template(conversation)))
        
        return {
            "conversations": conversations,
            "generated_token_counts": generated_token_counts
        }
    
    cols = dataset.column_names
    dataset = dataset.map(format_dataset,
                          batched=True, 
                          batch_size=len(dataset),
                          num_proc=4,
                          remove_columns=cols
                          )
    
    dataset.push_to_hub(f"{username}/{final_dataset_name}", private=False)

if __name__ == "__main__":
    process_distill_sft_dataset(
        dataset_name="aochongoliverli/Qwen2.5-Math-1.5B-deepmath-hard-1800-steps-4096-reasoning-data",
        model_name="Qwen/Qwen2.5-Math-1.5B"
    )
