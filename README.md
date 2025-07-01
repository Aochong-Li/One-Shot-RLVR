<h1 align="center">⏳ RL on CountDown Task</h1>

[Oliver Li](https://github.com/Aochong-Li)
<p align="center">

# Instruction 


## Install Zero environment
```
cd rlvr
git checkout countdown
conda create -n zero python=3.10
conda activate zero
pip install -r requirement.txt
```

## Change to the directory
```bash
cd rlvr
git pull
git checkout countdown
conda activate zero
```

## Generate data
```bash 
mkdir -p ./data/train/countdown/
python ./examples/data_preprocess/countdown.py --local_dir ./data/train/countdown/
```


 python push_to_hf/experiments.py --project_name verl_rlvr --run_name Qwen2.5-Math-1.5B-dsr_sub
 ```

 <!-- ## Launch GRPO jobs on DeepMath
 ```bash
 python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/deepmath_4096_hard"
 bash scripts/train/deepmath/training_1.5b_deepmath-8a100.sh
 ```

 ## Llama-Factory Installation
 ```bash
 cd sft/LLaMA-Factory
 conda deactivate
 conda create -n llama-factory python=3.11
 conda activate llama-factory
 
 pip install --upgrade huggingface_hub
 huggingface-cli login
 pip install nvitop
 pip install deepspeed==0.15.4
 pip install wandb
 pip install -e ".[torch,metrics]"
 ```

### Beta Run
 ```bash
 bash scripts/train/qwen25_math_deepmath_104k_4a100.sh
 ```

 ### Playground 
 ```bash
 input_ids = dataset_module['train_dataset'][0]['input_ids']
 labels = dataset_module['train_dataset'][0]['labels']
 attention_mask = dataset_module['train_dataset'][0]['attention_mask']
 ``` -->