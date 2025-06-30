from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

def process_sft_dataset(train_dataset_name_or_path: str,
                        hf_dataset_name: str
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

if __name__ == "__main__":
    process_sft_dataset(
        train_dataset_name_or_path="/share/goyal/lio/reasoning/data/countdown/sft/teacher_level3-4/Qwen3-4B.pickle",
        hf_dataset_name="Qwen3-4B-countdown-level3-4-sft-distill"
    )
