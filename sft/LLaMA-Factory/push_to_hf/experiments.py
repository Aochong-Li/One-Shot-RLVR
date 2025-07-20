from push import *
import os
import argparse

# Push model checkpoints to huggingface
def push_model_checkpoints (project_name: str, run_name: str):
    if os.path.exists(f"./outputs/{project_name}/{run_name}"):
        local_checkpoint_dir = f"./outputs/{project_name}/{run_name}"
    else:
        raise ValueError(f"Experiment {project_name}/{run_name} not found")

    global_steps = [step for step in os.listdir(local_checkpoint_dir) if "checkpoint-" in step]
    system_prompt = r"Please reason step by step and enclose your reasoning process within <think> </think> tags and put the final answer inside \\boxed{} tag."

    for global_step in global_steps:
        model_name = f"{run_name}-{global_step.replace('checkpoint-', 'step')}"
        # Use the checkpoint directory itself as it contains the model files
        local_checkpoint = f"{local_checkpoint_dir}/{global_step}"

        print("Pushing model checkpoint to huggingface: ", model_name)
        push_model_to_hf(model_name, local_checkpoint, new_system_prompt=system_prompt)

if __name__ == "__main__":
    """
    Example Usage:
    python3 push_to_hf/experiments.py --project_name deepmath --run_name Qwen2.5-1.5B-DeepMath-level1-5-117k-sft-5epochs-5e-5lr
    """

    args = argparse.ArgumentParser()
    args.add_argument("--project_name", type=str, required=True)
    args.add_argument("--run_name", type=str, required=True)
    args = args.parse_args()
    import pdb; pdb.set_trace()
    push_model_checkpoints(args.project_name, args.run_name)