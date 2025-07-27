from push import *
import os
import argparse

# Push model checkpoints to huggingface
def push_model_checkpoints (project_name: str, run_name: str, steps: list[int]=None, if_upload_folder: bool=False):
    if os.path.exists(f"./outputs/{project_name}/{run_name}"):
        local_checkpoint_dir = f"./outputs/{project_name}/{run_name}"
    else:
        raise ValueError(f"Experiment {project_name}/{run_name} not found")
    
    global_steps = [step for step in os.listdir(local_checkpoint_dir) if "global_step" in step]
    if steps is not None:
        global_steps = [step for step in global_steps if int(step.replace('global_step_', '')) in steps]

    for global_step in global_steps:
        model_name = f"{run_name}-{global_step.replace('global_step_', 'step')}"
        local_checkpoint = f"{local_checkpoint_dir}/{global_step}/actor"

        print("Pushing model checkpoint to huggingface: ", model_name)
        push_model_to_hf(model_name, local_checkpoint, if_upload_folder=if_upload_folder)

if __name__ == "__main__":
    """
    Example Usage:
    python3 push_to_hf/experiments.py \
        --project_name math8k \
        --run_name Qwen2.5-3B-math8k-sft-distill-150steps-dapo-10epochs-8rollouts-8192max-len \
        --steps 120 \
        --if_upload_folder
    """
    args = argparse.ArgumentParser()
    args.add_argument("--project_name", type=str, required=True)
    args.add_argument("--run_name", type=str, required=True)
    args.add_argument("--steps", nargs="+", type=int, required=False)
    args.add_argument("--if_upload_folder", action="store_true", required=False)
    args = args.parse_args()
    
    push_model_checkpoints(args.project_name, args.run_name, args.steps, args.if_upload_folder)