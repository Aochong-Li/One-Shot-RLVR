from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer

def process_sft_dataset(train_dataset_name_or_path: str,
                        hf_dataset_name: str,
                        **kwargs
                        ):
    df = pd.read_pickle(train_dataset_name_or_path)
    df = df[df['correct'] == 1.0].reset_index(drop=True)

    def format_dataset(row):
        problem = row['problem']
        response = row['response']
        conversations = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": response}
        ]

        return conversations
    
    df['conversations'] = df.apply(format_dataset, axis=1)
    df = df[['nums', 'target', 'level', 'conversations']]
    dataset = Dataset.from_pandas(df)

    dataset.push_to_hub(f"{hf_dataset_name}-sft-distill", private=False)

def process_corrupt_numbers_sft_dataset(train_dataset_name_or_path: str,
                                        tokenizer_name: str,
                                        hf_dataset_name: str
                                        ):
    """
    Generate RFT dataset from corrupt number reasoning traces.
    """
    df = pd.read_pickle(train_dataset_name_or_path)
    df = df[df['still_correct'] == 1.0].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    template_prefix, template_suffix = tokenizer.apply_chat_template([{"role":"user", "content": "HANDLE"}], tokenize=False, add_generation_prompt=True).split("HANDLE")

    def format_dataset(row):
        prompt = row['prompt']
        response = row['post_corruption_response']

        user_content, assistant_content = prompt.split(template_suffix)
        user_content = user_content.replace(template_prefix, '')
        assistant_content += response

        conversations = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        return conversations
    df['conversations'] = df.apply(format_dataset, axis=1)
    df = df[['nums', 'target', 'level', 'conversations']]
    dataset = Dataset.from_pandas(df)

    dataset.push_to_hub(hf_dataset_name, private=False)
    

if __name__ == "__main__":
    process_corrupt_numbers_sft_dataset(
        train_dataset_name_or_path="../perturb-r/results/countdown_train_stage2_level5_100K/corrupt_numbers/Qwen2.5-3B-countdown-level4-5-stage1_rl.pickle",
        hf_dataset_name="Qwen2.5-3B-grpo-stage1-countdown-level5-corrupt-numbers-sft-distill",
        tokenizer_name="aochongoliverli/Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch"
    )
