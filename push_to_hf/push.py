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
    python push_to_hf/push.py --model_name "SkyMath_8k_Qwen2.5_1.5B_sft_checkpoints64" --local_checkpoint_dir "/share/goyal/lio/reasoning/model/sky_math_8k/sft/Qwen2.5_Math_1.5B_sky_math8k_max_length_4096_bsz_32_epochs_10/checkpoint-64"
    python push_to_hf/push.py --model_name "Qwen2.5-3B-countdown-level4-5-grpo-20k-1epoch" --local_checkpoint_dir "./outputs/countdown/Qwen2.5-3B-sft-1epoch-countdown-level4-5-1epochs-4rollouts-4096max-length/global_step_156/actor"
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
