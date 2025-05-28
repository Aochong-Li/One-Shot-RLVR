import datasets
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, load_from_disk
import os
import sys
import json
import pandas as pd
from transformers import AutoTokenizer
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
import pdb
from tqdm import tqdm

"""
    Process OpenThoughts-114k-math dataset
    Process OpenR1-Math-220k dataset
    Combine OpenThoughts and OpenR1 datasets
    Save the combined dataset to a json file
"""
def extract_solution(solution_str):
    answer = last_boxed_only_string(solution_str)
    if answer is None:
        return None
    else:
        return remove_boxed(answer)

def process_openthought(example, tokenizer):
    source = example.pop("source")
    problem = example.pop("problem")
    solution = example.pop("solution")
    conversations = example.pop("conversations")
    generation = conversations[-1]['value'] \
        .replace("<|begin_of_thought|>", "") \
        .replace("<|end_of_thought|>", "") \
        .replace("<|begin_of_solution|>", "") \
        .replace("<|end_of_solution|>", "") \
        .strip("\n")
    problem_token_count = len(tokenizer.encode(problem))
    generated_token_count = example.pop("generated_token_count")

    answer = []
    answer_from_solution = extract_solution(solution)
    if answer_from_solution is not None:
        answer.append(answer_from_solution)
    answer_from_generation = extract_solution(generation)
    if answer_from_generation is not None:
        answer.append(answer_from_generation)

    return {
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "generation": generation,
        "source": source,
        "dataset": "OpenThoughts-114k-math",
        "generated_token_count": generated_token_count,
        "problem_token_count": problem_token_count,
    }

def process_openr1(example, tokenizer):
    source = example.pop("source")
    problem = example.pop("problem")
    solution = example.pop("solution")
    answer = example.pop("answer")
    generations = example.pop("generations")
    is_correct = example.pop("correctness_math_verify")
    
    generation, generation_len = None, float('inf')
    
    for i in range(len(is_correct)):
        if is_correct[i] and len(generations[i]) < generation_len:
            generation = generations[i]
            generation_len = len(generation)
    
    problem_token_count = len(tokenizer.encode(problem))
    if generation is not None:
        generation = generation.replace("<think>","").replace("</think>","")
        generated_token_count = len(tokenizer.encode(generation))
    else:
        generated_token_count = 0

    return {
        "problem": problem,
        "solution": solution,
        "answer": [answer],
        "generation": generation,
        "source": source,
        "dataset": "OpenR1-Math-220k",
        "generated_token_count": generated_token_count,
        "problem_token_count": problem_token_count,
    }

def deduplicate_combine_datasets(dataset1, dataset2):
    dataset1_problem_set = '\n'.join(dataset1['problem'])
    filtered_dataset2 = dataset2.filter(lambda x: x['problem'] not in dataset1_problem_set)
    
    return concatenate_datasets([dataset1, filtered_dataset2])

def process_super_rl_dataset(max_problem_token_len:int = 1024,
                             max_generated_token_len:int = 3072,
                             model_name = "Qwen/Qwen2.5-Math-1.5B",
                             local_output_dir:str = None,
                             username:str = "aochongoliverli",
                             dataset_name:str = None,
                             ):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load OpenThoughts-114k-math dataset, OpenR1 dataset
    openthought = load_dataset("bethgelab/CuratedThoughts", "OpenThoughts-114k-math-default")['train']
    openthought = openthought.filter(lambda x: x['correct'] and x['generated_token_count'] <= max_generated_token_len)
    openthought = openthought.map(lambda x: process_openthought(x, tokenizer), num_proc=4, remove_columns = ["messages", "system", "correct" ])
    openthought = openthought.filter(lambda x: len(x['answer']) > 0 and x['problem_token_count'] <= max_problem_token_len)

    # Load OpenR1 dataset
    openr1 = load_dataset("bethgelab/CuratedThoughts", "OpenR1-Math-220k-default")['train']
    openr1 = openr1.map(lambda x: process_openr1(x, tokenizer), num_proc=4, remove_columns = ["problem_type", "question_type", "uuid", "is_reasoning_complete", "correctness_llama", "finish_reasons", "correctness_count", "messages"])
    openr1 = openr1.filter(lambda x: x['generation'] is not None \
                           and x['generated_token_count'] <= max_generated_token_len \
                           and x['problem_token_count'] <= max_problem_token_len)

    super_rl = deduplicate_combine_datasets(openr1, openthought)    
    if local_output_dir is not None:
        super_rl.save_to_disk(local_output_dir)
    if dataset_name is not None:
        repo_id = f"{username}/{dataset_name}"
        super_rl.push_to_hub(repo_id, private=False)

"""
Process DeepMath dataset
"""

def process_deepmath_rl(example, tokenizer):
    problem = example.pop("question")
    answer = example.pop("final_answer")
    generations = [example.pop("r1_solution_1"), example.pop("r1_solution_2"), example.pop("r1_solution_3")]
    generation = generations[0]
    
    for example in generations:
        if len(example) < len(generation):
            generation = example
    
    problem_token_count = len(tokenizer.encode(problem))
    generated_token_count = len(tokenizer.encode(generation))

    return {
        "problem": problem,
        "answer": answer,
        "generation": generation,
        "generated_token_count": generated_token_count,
        "problem_token_count": problem_token_count,
        "dataset": "DeepMath-4096",
    }

def process_deepmath_sft(batch, tokenizer, min_difficulty_level):
    problems, answers, difficulties, generations, problem_token_counts, generated_token_counts = [], [], [], [], [], []
    # To get a proper progress bar, use tqdm on a range and index into the batch arrays.
    n = len(batch['question'])
    for i in tqdm(range(n), desc="Processing DeepMath dataset", leave=True):
        question = batch['question'][i]
        answer = batch['final_answer'][i]
        difficulty = batch['difficulty'][i]
        r1_solution_1 = batch['r1_solution_1'][i].replace("<think>","").replace("</think>","")
        r1_solution_2 = batch['r1_solution_2'][i].replace("<think>","").replace("</think>","")
        r1_solution_3 = batch['r1_solution_3'][i].replace("<think>","").replace("</think>","")

        if difficulty >= min_difficulty_level:
            problems.extend([question] * 3)
            answers.extend([answer] * 3)
            difficulties.extend([difficulty] * 3)
            generations.extend([r1_solution_1, r1_solution_2, r1_solution_3])

            problem_token_counts.extend([len(tokenizer.encode(question))] * 3)
            generated_token_counts.extend([
                len(tokenizer.encode(r1_solution_1)),
                len(tokenizer.encode(r1_solution_2)),
                len(tokenizer.encode(r1_solution_3))
            ])

    return {
        "problem": problems,
        'answer': answers,
        "difficulty": difficulties,
        "generation": generations,
        "problem_token_count": problem_token_counts,
        "generated_token_count": generated_token_counts,
        "dataset": ["DeepMath-4096"] * len(problems)
    }

def convert_to_conversation_format(example):
    problem = example.pop("problem")
    generation = example.pop("generation")
    conversations = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": generation}
    ]
    return {
        "conversations": conversations
    }

def process_deepmath_dataset(max_problem_token_len:int = 200,
                              max_generated_token_len:int = 3840,
                              model_name = "Qwen/Qwen2.5-Math-1.5B",
                              local_output_dir:str = None,
                              username:str = "aochongoliverli",
                              dataset_name:str = None,
                              min_difficulty_level: float = 0.0
                              ):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    deepmath_dataset = load_dataset("zwhe99/DeepMath-103K")['train']
    
    # SFT dataset for Distillation
    columns = ["question", "final_answer", "topic", "r1_solution_1", "r1_solution_2", "r1_solution_3"]
    deepmath_sft_dataset = deepmath_dataset.map(lambda x: process_deepmath_sft(x, tokenizer, min_difficulty_level),
                                                batched=True,
                                                batch_size=len(deepmath_dataset),
                                                num_proc=4,
                                                remove_columns = columns)
    
    deepmath_sft_dataset = deepmath_sft_dataset.filter(lambda x: x['generated_token_count'] <= max_generated_token_len \
                                                and x['problem_token_count'] <= max_problem_token_len)
    # RL dataset for GRPO
    deepmath_rl_dataset = deepmath_sft_dataset.select_columns(["problem", "answer", "difficulty", "problem_token_count", "dataset"])
    deepmath_rl_dataset = pd.DataFrame(deepmath_rl_dataset).drop_duplicates(subset=['problem', 'answer'])
    deepmath_rl_dataset = Dataset.from_pandas(deepmath_rl_dataset, preserve_index=False)
    
    # Convert SFT datasetto conversation format
    deepmath_sft_dataset = deepmath_sft_dataset.map(convert_to_conversation_format,
                                                    num_proc=4,
                                                    remove_columns=["problem", "answer", "difficulty", "generation"]
                                                    )
    if local_output_dir is not None:
        deepmath_rl_dataset.save_to_disk(os.path.join(local_output_dir, "rl"))
        deepmath_sft_dataset.save_to_disk(os.path.join(local_output_dir, "sft"))
    if dataset_name is not None:
        repo_id_rl = f"{username}/{dataset_name}-rl"
        repo_id_sft = f"{username}/{dataset_name}-sft"

        deepmath_rl_dataset.push_to_hub(repo_id_rl, private=False)
        deepmath_sft_dataset.push_to_hub(repo_id_sft, private=False)

if __name__ == "__main__":
    process_deepmath_dataset(model_name="Qwen/Qwen2.5-Math-1.5B", dataset_name="deepmath-4096-hard", min_difficulty_level=5.0)