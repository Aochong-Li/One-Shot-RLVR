import random
from fractions import Fraction
from typing import List, Tuple, Sequence
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from tqdm.auto import tqdm
import argparse

def generate_countdown_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: Sequence[str] = ('+', '-', '*', '/', '//', '%'),
    mandatory_operations: Sequence[str] = (),
    seed: int = 42,
    prior_nums: set[Tuple[int, ...]] = set()
) -> Dataset:
    """
    Return a 🤗 Datasets object with `num_samples` rows, guaranteed solvable.
    Each row = (nums: List[int], target: int, solution: str).

    * Uses tqdm to show generation progress.
    * Ensures rows are unique by the `nums` list.
    """
    rng        = random.Random(seed)
    rows       : list[Tuple[List[int], int, str]] = []
    seen_nums  : set[Tuple[int, ...]] = set()

    # ------------------------- helper ------------------------------------
    def combine(a_val, a_exp, b_val, b_exp, op):
        if op == '+':
            return a_val + b_val, f"({a_exp}+{b_exp})"
        if op == '-':
            return a_val - b_val, f"({a_exp}-{b_exp})"
        if op == '*':
            return a_val * b_val, f"({a_exp}*{b_exp})"
        if op == '/':
            if b_val == 0 or a_val % b_val != 0:
                return None
            return a_val / b_val, f"({a_exp}/{b_exp})"          # exact Fraction
        if op == '//':                                          # floor-div
            if b_val == 0:
                return None
            val = Fraction(a_val // b_val)                      # keep Fraction
            return val, f"({a_exp}//{b_exp})"
        if op == '%':                                           # modulo
            if b_val == 0:
                return None
            val = Fraction(a_val % b_val)
            return val, f"({a_exp}%{b_exp})"
        raise ValueError(f"Unknown op {op!r}")

    progress_bar = tqdm(total=num_samples,
                        desc=f"Generating dataset for level {num_operands}",
                        unit="sample")

    while len(rows) < num_samples:                  # ← use rows, not dataset
        nums = [rng.randint(min_number, max_number) for _ in range(num_operands)]
        tup_nums = tuple(nums)
        if tup_nums in seen_nums:
            continue

        pool = [(Fraction(n), str(n)) for n in nums]     # (value, expression)

        while len(pool) > 1:
            ia, ib          = rng.sample(range(len(pool)), 2)
            a_val, a_exp    = pool.pop(max(ia, ib))
            b_val, b_exp    = pool.pop(min(ia, ib))
            op              = rng.choice(operations)

            res = combine(a_val, a_exp, b_val, b_exp, op)
            retry = 0
            while res is None and retry < 10:
                op  = rng.choice(operations)
                res = combine(a_val, a_exp, b_val, b_exp, op)
                retry += 1
            if res is None:          # failed to merge ⇒ discard sample
                break
            pool.append(res)

        else:  # executed only if the inner while didn't `break`
            expr = pool[0][1]
            if mandatory_operations and not any(mop in expr for mop in mandatory_operations):
                continue

            target = pool[0][0]
            if target.denominator == 1 and 0 < target <= max_target:
                rows.append(
                    {
                        "nums": nums,
                        "target": int(target),
                        "solution": pool[0][1]
                    }
                )
                seen_nums.add(tup_nums)
                progress_bar.update(1)

    progress_bar.close()
    return Dataset.from_list(rows)

if __name__ == "__main__":
    """
    python data/generate_dataset/generate_countdown.py --levels 5 6 --train_size 100000 --test_size 5000 --ood_test_size 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--train_size", type=int, default=45000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--ood_test_size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    hf_username = "aochongoliverli"

    for level in args.levels:
        if not args.overwrite:
            try:
                prior_dataset = load_dataset(f"{hf_username}/countdown_level_{args.levels[0]}")
                prior_nums = set(tuple(row['nums']) for row in prior_dataset['train']).union(set(tuple(row['nums']) for row in prior_dataset['test']))
            except:
                prior_nums = set()
                print(f"This is the first time generating the dataset for level {level}")

        if args.train_size + args.test_size > 0:
            dataset = generate_countdown_dataset(args.train_size + args.test_size,
                                                num_operands=level,
                                                max_target=1000,
                                                min_number=1,
                                                max_number=100,
                                                operations=['+', '-', '*', '/'],
                                                prior_nums=prior_nums
                                                )
            dataset = dataset.train_test_split(test_size=args.test_size)
            if prior_nums:
                for split, data in prior_dataset.items():
                    if split in dataset:
                        dataset[split] = concatenate_datasets([data, dataset[split]])
                    else:
                        dataset[split] = data
                    
        if args.ood_test_size > 0:
            ood_dataset = generate_countdown_dataset(args.ood_test_size,
                                                    num_operands=level,
                                                    max_target=1000,
                                                    min_number=1,
                                                    max_number=100,
                                                    operations=['+', '-', '*', '/', '//', '%'],
                                                    mandatory_operations=['//', '%']
                                                    )
            if 'ood_test' in dataset:
                dataset['ood_test'] = concatenate_datasets([dataset['ood_test'], ood_dataset])
            else:
                dataset['ood_test'] = ood_dataset
        
        dataset.push_to_hub(f"{hf_username}/countdown_level_{level}")



