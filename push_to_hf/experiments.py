from push import *
import os
import glob
import argparse

# Push model checkpoints to huggingface
def push_model_checkpoints (experiment_name: str):
    local_checkpoint_dir = f"./outputs/verl_rlvr/{experiment_name}"
    global_steps = [step for step in os.listdir(local_checkpoint_dir) if "global_step" in step]

    for global_step in global_steps:
        model_name = f"{experiment_name}-{global_step}"
        local_checkpoint = f"{local_checkpoint_dir}/{global_step}/actor"
        try:
            for filepath in glob.glob(os.path.join(local_checkpoint, "*.pt")):
                print(f"Removing optimizer state: {filepath}")
                os.remove(filepath)
        except Exception as e:
            print(f"Error removing optimizer state: {e}")

        print("Pushing model checkpoint to huggingface: ", model_name)
        push_model_to_hf(model_name, local_checkpoint)

if __name__ == "__main__":
    """
    Example Usage:
    python3 push_to_hf/experiments.py --experiment_name Qwen2.5-Math-1.5B-deepmath-g2-entropy-0.003-beta
    """
    args = argparse.ArgumentParser()
    args.add_argument("--experiment_name", type=str, required=True)
    args = args.parse_args()

    push_model_checkpoints(args.experiment_name)