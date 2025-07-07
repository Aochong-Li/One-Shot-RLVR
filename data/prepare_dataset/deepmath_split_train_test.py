from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, ClassLabel
from transformers import AutoTokenizer

def uniform_topic_sampling (dataset: Dataset, test_size: int):
    if len(dataset) == 0:
        return None

    if dataset['difficulty'][0] < 3.0:
        return DatasetDict({
            "train": dataset,
            "test":  Dataset.from_dict({})
        })
    if dataset['difficulty'][0] > 9.0:
        return DatasetDict({
            "train": Dataset.from_dict({}),
            "test":  dataset
        })

    return dataset.train_test_split(
                test_size = test_size,
                seed = 42,
                stratify_by_column = "subtopic")

def extract_subtopic (topic: str):
    parts = [p.strip() for p in topic.split("->")]
    
    if parts and parts[0].lower() == "mathematics":
        parts = parts[1:]
    
    return parts[0] if parts else topic

def prompt_length(prompt: str, tokenizer: AutoTokenizer):
    return len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))

def split_deepmath_dataset(tokenizer_name: str, max_prompt_length: int = 512, test_size_per_level = 200, new_dataset_name = "aochongoliverli/DeepMath-103K"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_dataset("zwhe99/DeepMath-103K")["train"]
    dataset = dataset.map(lambda x: {"subtopic": extract_subtopic(x["topic"])}, num_proc = 5)
    if not isinstance(dataset.features["subtopic"], ClassLabel):
        dataset = dataset.class_encode_column("subtopic")

    level_splits = {}
    levels = list(set(dataset["difficulty"]))
    levels = [level for level in levels if level >= 0.0]
    levels.sort()
    
    for level in levels:
        level_dataset = dataset.filter(lambda x: x["difficulty"] == level, num_proc = 5)
        if level >= 5:
            # For level 5 and above, we remove T/F Yes/No multiple choice questions to make guessing the answer harder
            level_dataset = level_dataset.filter(lambda x: x["final_answer"].lower() not in ["true", "false", "yes", "no", 'a', 'b', 'c', 'd', 'e'])
            level_dataset = level_dataset.filter(lambda x: prompt_length(x["question"], tokenizer) <= max_prompt_length)
        
        if level_dataset == None or len(level_dataset) == 0:
            continue
        
        level_splits[level] = uniform_topic_sampling(level_dataset, test_size_per_level)

    train_sets = [d["train"] for d in level_splits.values()]
    test_sets  = [d["test"]  for d in level_splits.values()]

    final = DatasetDict({
        "train": concatenate_datasets(train_sets),
        "test":  concatenate_datasets(test_sets),
    })    

    final.push_to_hub(new_dataset_name)

if __name__ == "__main__":
    split_deepmath_dataset(tokenizer_name = "Qwen/Qwen2.5-1.5B", test_size_per_level = 200, new_dataset_name = "aochongoliverli/DeepMath-103K")

