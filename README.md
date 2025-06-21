<div align="center">

# Reinforcement Learning for Reasoning in Large Language Models with One Training Example


[Oliver Li](https://github.com/Aochong-Li)

</div>

<!-- ## Updates
* 17/05/2025: We release our [checkpoints](https://huggingface.co/collections/ypwang61/one-shot-rlvr-6827f72c3359b2ffe75fc1a8) and [dataset](https://huggingface.co/datasets/ypwang61/one_shot_rlvr) in huggingface.
* 30/04/2025: 🎉 We release our [paper](https://arxiv.org/abs/2504.20571), [code](https://github.com/ypwang61/One-Shot-RLVR), and [wandb records](https://wandb.ai/yipingwanguw/verl_few_shot?nw=nwuseryipingwang22). See the summarization of our work at [X(twitter)](https://x.com/ypwang61/status/1917596101953348000). -->

# Instruction 

## Change to the directory
```bash
cd rlvr
git pull
git checkout grpo
conda activate zero
```

<!-- ## Upload model weights
```bash
 python push_to_hf/experiments.py --project_name verl_rlvr --run_name Qwen2.5-Math-1.5B-dsr_sub
 ``` -->

 ## Generate DeepMath Training Data
 ```bash
 python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/" --difficulty_levels 5 6
 python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/" --difficulty_levels 7 8 9
 ```

 ## Launch GRPO jobs on DeepMath
 ```bash
 python data/format_parquet/deepmath_dataset.py --local_dir "./data/train/deepmath_4096_hard"
 bash scripts/train/deepmath/training_1.5b_deepmath-8a100.sh
 ```

 ## Verl Environment
 ```bash
 conda create -y -n zero python=3.9
 conda activate zero
 cd rlvr
 git pull
 pip install -r requirements.txt 
 ```  

 ## Evaluation Environment
 ```bash
 conda create -y -n rlvr_eval python=3.10
 conda activate rlvr_eval
 cd Qwen2.5-Eval/evaluation
 cd latex2sympy
 pip install -e .
 cd ..
 pip install -r requirements.txt 
 pip install vllm==0.5.1 --no-build-isolation
 pip install transformers==4.42.3
 pip install wandb matplotlib
 pip install -U transformers
 pip install vllm==0.6.3
 pip install vllm-flash-attn==2.6.2
 ```        

 ## Llama-Factory Installation
 ```bash
 cd sft/LLaMA-Factory
 conda deactivate
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
 ```


