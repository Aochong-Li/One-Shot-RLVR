import datasets
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, load_from_disk
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
import pdb

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

def process_super_rl_dataset(output_dir:str, max_problem_token_len:int = 1024, max_generated_token_len:int = 3072, model_name = "Qwen/Qwen2.5-Math-1.5B"):
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
    super_rl.save_to_disk(output_dir)

"""
Process DeepMath dataset
"""
def process_deepmath(example, tokenizer):
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
        "dataset": "DeepMath-103k",
    }

def process_deepmath_dataset(output_dir:str, max_problem_token_len:int = 200, max_generated_token_len:int = 3840, model_name = "Qwen/Qwen2.5-Math-1.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    deepmath_dataset = load_dataset("zwhe99/DeepMath-103K")['train']
    deepmath_dataset = deepmath_dataset.map(lambda x: process_deepmath(x, tokenizer), num_proc=4, remove_columns = ["r1_solution_1", "r1_solution_2", "r1_solution_3", "final_answer", "question"])
    deepmath_dataset = deepmath_dataset.filter(lambda x: x['generated_token_count'] <= max_generated_token_len \
                                                and x['problem_token_count'] <= max_problem_token_len)

    deepmath_dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    dataset_dir = "/share/goyal/lio/reasoning/data/deepmath_4096"
    process_deepmath_dataset(output_dir=dataset_dir, model_name="Qwen/Qwen2.5-Math-1.5B")