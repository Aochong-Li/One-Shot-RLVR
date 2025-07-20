from huggingface_hub import HfApi, upload_folder
from datasets import Dataset, DatasetDict, load_from_disk
import argparse
from transformers import AutoTokenizer

api = HfApi()
USERNAME = "aochongoliverli"

def push_model_to_hf(model_name: str, local_checkpoint_dir: str, username: str=USERNAME, new_system_prompt: str=None):
    repo_id = f"{username}/{model_name}"
    
    # Check if repo already exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Repository {repo_id} already exists, skipping creation.")
    except Exception:
        print(f"Creating new repository {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)

    tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_dir)
    tokenizer.name_or_path = repo_id
    tokenizer.save_pretrained(repo_id)

    if new_system_prompt is not None:
        old_template = tokenizer.chat_template
        new_template = old_template.replace("You are a helpful assistant.", new_system_prompt)
        tokenizer.chat_template = new_template
        tokenizer.save_pretrained(repo_id)
    
    # Upload folder to the repository with progress tracking
    upload_folder(
        folder_path=local_checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )

def push_dataset_to_hf(dataset_name: str, local_dataset_dir: str, username: str=USERNAME):
    dataset = load_from_disk(local_dataset_dir)
    
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})
    elif isinstance(dataset, DatasetDict):
        pass
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
    
    repo_id = f"{username}/{dataset_name}"
    dataset.push_to_hub(repo_id, private=False)

if __name__ == "__main__":
    """
    Example usage:
    python push_to_hf/push.py --model_name "Qwen2.5-1.5B-DeepMath-level5-grpo-initial-checkpoint" --local_checkpoint_dir "outputs/deepmath/Qwen2.5-1.5B-DeepMath-level1-5-117k-sft-5epochs-5e-5lr/checkpoint-4570"
    python push_to_hf/push.py --dataset_name "deepmath_4096" --local_dataset_dir "/share/goyal/lio/reasoning/data/deepmath_4096"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--local_checkpoint_dir", type=str, required=False)
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--local_dataset_dir", type=str, required=False)    
    args = parser.parse_args()
    
    system_prompt = r"Please reason step by step and enclose your reasoning process within <think> </think> tags and put the final answer inside \\boxed{} tag."
    if args.model_name is not None and args.local_checkpoint_dir is not None:
        push_model_to_hf(args.model_name, args.local_checkpoint_dir, new_system_prompt=system_prompt)
    if args.dataset_name is not None and args.local_dataset_dir is not None:
        push_dataset_to_hf(args.dataset_name, args.local_dataset_dir)
