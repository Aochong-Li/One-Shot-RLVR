from push import *

local_checkpoint_dir = "./outputs/Qwen2.5-Math-1.5B-dsr_sub/actor"
epochs = [4, 9, 14, 19]

for epoch in epochs:
    model_name = f"Qwen2.5-Math-1.5B_drgrpo_parquet_rollout_8_max_length_3000_epoch_{epoch}"
    local_checkpoint = f"{local_checkpoint_dir}/epoch_{epoch}"
    print(local_checkpoint)
    push_model_to_hf(model_name, local_checkpoint)