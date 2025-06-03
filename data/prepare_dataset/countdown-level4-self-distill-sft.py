from datasets import load_dataset, Dataset, DatasetDict
import os
import sys
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def process_rollout_sft_dataset(train_dataset_name_or_path: str,
                                eval_dataset_name_or_path: str=None,
                                model_name: str="Qwen/Qwen2.5-3B",
                                ):
    dataset = load_dataset(train_dataset_name_or_path)['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def format_dataset(batch, tokenizer = tokenizer):
        indices, conversations, rewards, global_steps, generated_token_counts = [], [], [], [], []
        n = len(batch['question'])

        for i in tqdm(range(n), desc="Formatting Dataset for SFT", leave=True):
            index = batch['index'][i]
            question = batch['question'][i]
            response = batch['response'][i]
            reward = batch['reward'][i]
            global_step = batch['global_step'][i]
            
            for j in range(len(response)):
                conversation = [
                        {"role": "user", "content": question.replace(tokenizer.eos_token, "")},
                        {"role": "assistant", "content": response[j].replace(tokenizer.eos_token, "")}
                    ]
                
                indices.append(index)
                conversations.append(conversation)
                rewards.append(reward[j])
                global_steps.append(global_step[j])
                generated_token_counts.append(len(tokenizer.encode(response[j]))) # Only response tokens are used for loss
                
        
        return {
            "index": indices,
            "conversations": conversations,
            "rewards": rewards,
            "global_steps": global_steps,
            "generated_token_counts": generated_token_counts
        }
    
    cols = dataset.column_names
    dataset = dataset.map(format_dataset,
                          batched=True, 
                          batch_size=len(dataset),
                          num_proc=6,
                          remove_columns=cols
                          )
    success_attempts = dataset.filter(lambda x: x["rewards"] == 1.)
    correct_format_attempts = dataset.filter(lambda x: x['rewards'] > 0.)

    hf_datadict = DatasetDict({
        "correct": success_attempts,
        "correct_format": correct_format_attempts,
        "raw": dataset
    })

    hf_datadict.push_to_hub(f"{train_dataset_name_or_path}-rollout-sft", private=False)

if __name__ == "__main__":
    process_rollout_sft_dataset(
        train_dataset_name_or_path="aochongoliverli/Qwen2.5-3B-countdown-level4-1epochs-4rollouts-1024max-length-reasoning-traces",
        model_name="Qwen/Qwen2.5-3B"
    )
