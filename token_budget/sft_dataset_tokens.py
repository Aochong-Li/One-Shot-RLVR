from datasets import load_dataset
from transformers import AutoTokenizer

sft_dataset = load_dataset("aochongoliverli/deepmath-4096-hard-sft")["train"]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

def count_tokens (example, tokenizer):
    conversations = example["conversations"]
    assistant = conversations[1]["content"]
    num_tokens = len(tokenizer.encode(assistant))
    return {'num_tokens': num_tokens}

sft_dataset = sft_dataset.map(lambda x: count_tokens(x, tokenizer), num_proc=4)

print(sft_dataset[0])
