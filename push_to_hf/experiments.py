from push import *
import os
import argparse

# Push model checkpoints to huggingface
def push_model_checkpoints (root_dir: str, project_name: str, run_name: str, steps: list[int]=None, if_upload_folder: bool=False):
    if os.path.exists(f"{root_dir}/{project_name}/{run_name}"):
        local_checkpoint_dir = f"{root_dir}/{project_name}/{run_name}"
    else:
        raise ValueError(f"Experiment {root_dir}/{project_name}/{run_name} not found")
    
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
        --root_dir /share/goyal/lio/reasoning/model/sft \
        --project_name math8k \
        --run_name Qwen2.5-3B-math8k-QwQ-400steps-dapo-5epochs-8rollouts-16384max-len \
        --steps 20 40 60 80 100 120
    """
    args = argparse.ArgumentParser()
    args.add_argument("--root_dir", type=str, required=True)
    args.add_argument("--project_name", type=str, required=True)
    args.add_argument("--run_name", type=str, required=True)
    args.add_argument("--steps", nargs="+", type=int, required=False)
    args.add_argument("--if_upload_folder", action="store_true", required=False)
    args = args.parse_args()
    
    push_model_checkpoints(args.root_dir, args.project_name, args.run_name, args.steps, args.if_upload_folder)