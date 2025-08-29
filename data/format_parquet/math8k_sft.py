from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer
import random

def process_sft_dataset(sft_filepath: str,
                        hf_dataset_name: str,
                        correct_col: str = "is_correct",
                        **kwargs
                        ):
    import pdb; pdb.set_trace()
    sft_df = pd.read_pickle(sft_filepath)
    sft_df = sft_df[(sft_df[correct_col] > 0.0) & (sft_df["if_boxed"])].reset_index(drop=True)
    
    def format_dataset(row):
        problem = row['problem']
        if "<think>" not in row['response']:
            response = "<think>\n" + row['response']
        else:
            response = row['response']
        
        conversations = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": response}
        ]
        return conversations
    
    sft_df['conversations'] = sft_df.apply(format_dataset, axis=1)
    sft_df = sft_df[["problem", "solution", "level", "conversations"]]

    dataset = Dataset.from_pandas(sft_df)
    dataset.push_to_hub(hf_dataset_name, private=False)

def process_limo_dataset(tokenizer_name: str, max_tokens: int = 16384):
    limo_df = pd.DataFrame(load_dataset("GAIR/LIMO")["train"])
    qwq_df = pd.DataFrame(load_dataset("aochongoliverli/math8k-sft-QwQ-32B-16k-reasoning-traces")["train"]).drop(columns = ['level'])
    qwen3_df = pd.DataFrame(load_dataset("aochongoliverli/math8k-sft-Qwen3-32B-16k-reasoning-traces")["train"]).drop(columns = ['level'])
    am_df = pd.DataFrame(load_dataset("aochongoliverli/math8k-sft-AM-Distill-Qwen-32B-16k-reasoning-traces")["train"]).drop(columns = ['level'])

    def format_limo_dataset(example):
        question = example["question"]
        solution = example["solution"]

        if "<think>" not in solution:
            solution = "<think>\n" + solution
            
        conversations = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution}
        ]
        
        return conversations
    
    def count_tokens(conversations: list):
        return len(tokenizer.apply_chat_template(conversations))
    
    limo_df["conversations"] = limo_df.apply(format_limo_dataset, axis=1)
    limo_df = limo_df.drop(columns = ["solution"]).rename(columns={"question": "problem", "answer": "solution"})

    # Count tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    limo_df["num_tokens"] = limo_df["conversations"].apply(count_tokens)
    qwq_df["num_tokens"] = qwq_df["conversations"].apply(count_tokens)
    qwen3_df["num_tokens"] = qwen3_df["conversations"].apply(count_tokens)
    am_df["num_tokens"] = am_df["conversations"].apply(count_tokens)

    # Filter by max tokens
    limo_df = limo_df[limo_df["num_tokens"] <= max_tokens].reset_index(drop=True)
    qwq_df = qwq_df[qwq_df["num_tokens"] <= max_tokens]
    qwen3_df = qwen3_df[qwen3_df["num_tokens"] <= max_tokens]
    am_df = am_df[am_df["num_tokens"] <= max_tokens]

    # Add source column
    limo_df["source"] = "limo"
    qwq_df["source"] = "qwq-math8k"
    qwen3_df["source"] = "qwen3-math8k"
    am_df["source"] = "am-math8k"

    # Combine datasets
    qwq_df = pd.concat([qwq_df, limo_df], ignore_index=True)
    qwen3_df = pd.concat([qwen3_df, limo_df], ignore_index=True)
    am_df = pd.concat([am_df, limo_df], ignore_index=True)

    data_dict = DatasetDict({
        "limo": Dataset.from_pandas(limo_df),
        "math8kqwq": Dataset.from_pandas(qwq_df),
        "math8kqwen3": Dataset.from_pandas(qwen3_df),
        "math8kam": Dataset.from_pandas(am_df),
    })

    data_dict.push_to_hub("aochongoliverli/limo-math8k-distill-16k-reasoning-traces", private=False)
    

def process_coldstart_dataset(sft_hf_dataset_name: str,
                               dataset_size: int,
                               hf_dataset_name: str,
                               ):
    sft_data = load_dataset(sft_hf_dataset_name, split="train")
    sft_df = sft_data.to_pandas()
    cold_start_df = sft_df[sft_df["level"] == 3]
    
    subjects = cold_start_df["subject"].value_counts()
    inv_w = 1 / subjects
    inv_w = inv_w / inv_w.sum()

    cold_start_df = cold_start_df.sample(n=dataset_size, weights=cold_start_df["subject"].map(inv_w), random_state=42)
    cold_start_df = cold_start_df.reset_index(drop=True)
    
    dataset = Dataset.from_pandas(cold_start_df)
    dataset.push_to_hub(hf_dataset_name, private=False)
    
if __name__ == "__main__":
    
    process_limo_dataset(
        tokenizer_name="Qwen/Qwen2.5-3B",
        max_tokens=16384
    )
    
    # process_coldstart_dataset(
    #     sft_hf_dataset_name="aochongoliverli/math8k-sft-QwQ-32B-reasoning-traces",
    #     dataset_size=1500,
    #     hf_dataset_name="aochongoliverli/math8k-coldstart-QwQ-32B-reasoning-traces",
    # )