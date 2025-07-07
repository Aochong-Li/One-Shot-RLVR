from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer
import random

def process_sft_dataset(train_dataset_name_or_path: str,
                        hf_dataset_name: str,
                        min_level: int,
                        max_level: int,
                        dataset_size = None,
                        **kwargs
                        ):
    dataset = load_dataset(train_dataset_name_or_path)["train"]
    dataset = dataset.filter(lambda x: x['difficulty'] > min_level and x['difficulty'] < max_level, num_proc = 5)
    
    if dataset_size is not None:
        train_dataset, _ = dataset.train_test_split(
            test_size = len(dataset) - dataset_size, 
            stratify_by_column = "subtopic",
            seed = 42)
    else:
        train_dataset = dataset

    def format_dataset(row):
        problem = row['question']
        sol1, sol2, sol3 = row['r1_solution_1'], row['r1_solution_2'], row['r1_solution_3']
        response = random.choice([sol1, sol2, sol3])
        if "<think>" not in response:
            response = "<think>\n" + response

        conversations = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": response}
        ]
        return conversations

    train_df = train_dataset.to_pandas()
    train_df['conversations'] = train_df.apply(format_dataset, axis=1)
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.remove_columns(["r1_solution_1", "r1_solution_2", "r1_solution_3"])
    train_dataset.push_to_hub(hf_dataset_name, private=False)

if __name__ == "__main__":
    process_sft_dataset(
        train_dataset_name_or_path="aochongoliverli/DeepMath-103K",
        hf_dataset_name="aochongoliverli/DeepMath-level1-4-14k-sft-stage0",
        min_level=0,
        max_level=5,
    )
