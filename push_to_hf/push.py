from huggingface_hub import HfApi, upload_folder
from datasets import Dataset, DatasetDict, load_from_disk
import argparse

api = HfApi()
USERNAME = "aochongoliverli"

def push_model_to_hf(model_name: str, local_checkpoint_dir: str, username: str=USERNAME):
    repo_id = f"{username}/{model_name}"
    
    # Check if repo already exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Repository {repo_id} already exists, skipping creation.")
    except Exception:
        print(f"Creating new repository {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)
    
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
    python push_to_hf/push.py --model_name "R1-Distill-Qwen-1.5B-DeepMath-stage2-grpo-level3-4-5epochs-4rollouts-8192max-length-global_step_255" --local_checkpoint_dir "outputs/deepmath/R1-Distill-Qwen-1.5B-DeepMath-stage2-grpo-level3-4-5epochs-4rollouts-8192max-length/global_step_255/actor"
    python push_to_hf/push.py --dataset_name "deepmath_4096" --local_dataset_dir "/share/goyal/lio/reasoning/data/deepmath_4096"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--local_checkpoint_dir", type=str, required=False)
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--local_dataset_dir", type=str, required=False)
    
    args = parser.parse_args()
    
    if args.model_name is not None and args.local_checkpoint_dir is not None:
        push_model_to_hf(args.model_name, args.local_checkpoint_dir)
    if args.dataset_name is not None and args.local_dataset_dir is not None:
        push_dataset_to_hf(args.dataset_name, args.local_dataset_dir)
