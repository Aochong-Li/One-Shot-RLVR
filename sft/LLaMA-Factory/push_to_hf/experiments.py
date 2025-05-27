from push import *
import os
import argparse

# Push model checkpoints to huggingface
def push_model_checkpoints (project_name: str, run_name: str):
    if os.path.exists(f"./outputs/{project_name}/{run_name}"):
        local_checkpoint_dir = f"./outputs/{project_name}/{run_name}"
    elif os.path.exists(f"./outputs/{run_name}/{project_name}/{run_name}"):
        local_checkpoint_dir = f"./outputs/{run_name}/{project_name}/{run_name}"
    else:
        raise ValueError(f"Experiment {project_name}/{run_name} not found")

    global_steps = [step for step in os.listdir(local_checkpoint_dir) if "checkpoint-" in step]

    for global_step in global_steps:
        if global_step != "checkpoint-4890":
            continue
        model_name = f"{run_name}-{global_step.replace('checkpoint-', '')}"
        # Use the checkpoint directory itself as it contains the model files
        local_checkpoint = f"{local_checkpoint_dir}/{global_step}"

        print("Pushing model checkpoint to huggingface: ", model_name)
        push_model_to_hf(model_name, local_checkpoint)

if __name__ == "__main__":
    """
    Example Usage:
    python3 push_to_hf/experiments.py --project_name llamafactory-sft --run_name Qwen2.5-Math-1.5B-DeepMath-Hard-SFT
    """
    args = argparse.ArgumentParser()
    args.add_argument("--project_name", type=str, required=True)
    args.add_argument("--run_name", type=str, required=True)
    args = args.parse_args()

    push_model_checkpoints(args.project_name, args.run_name)