import ray
import os

if not ray.is_initialized():
    # this is for local ray cluster
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # Add num_gpus=0 to tell Ray not to touch the GPUs for this test
    ray.init(
        num_gpus=0,
        runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
        _temp_dir='/share/goyal/ray_tmp',
        )
    
    # If ray.init() succeeds, this line will be printed.
    print("ray initialized successfully!")