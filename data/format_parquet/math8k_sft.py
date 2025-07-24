from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer
import random

def process_sft_dataset(sft_filepath: str,
                        hf_dataset_name: str,
                        **kwargs
                        ):
    sft_df = pd.read_pickle(sft_filepath)
    sft_df = sft_df[sft_df["is_correct"] > 0.0].reset_index(drop=True)
    
    def format_dataset(row):
        problem = row['problem']
        response = "<think>\n" + row['response']
        
        conversations = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": response}
        ]
        return conversations
    
    sft_df['conversations'] = sft_df.apply(format_dataset, axis=1)
    sft_df = sft_df[["problem", "solution", "level", "subject", "conversations"]]

    dataset = Dataset.from_pandas(sft_df)
    dataset.push_to_hub(hf_dataset_name, private=False)

if __name__ == "__main__":
    process_sft_dataset(
        sft_filepath="/mnt/home/al2644/research/projects/perturb-r/results/math8k/benchmark/QwQ-32Btrain.pickle",
        hf_dataset_name="aochongoliverli/math8k-sft-QwQ-32B-reasoning-traces",
    )
