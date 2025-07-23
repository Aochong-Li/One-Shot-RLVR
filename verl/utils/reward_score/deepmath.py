import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.openmathinst_utils import extract_answer, math_equal
from deepscaler import compute_score as deepscaler_compute_score
from hendrycks_math import compute_score as hendrycks_math_compute_score

from math_verify import verify, parse
from typing import Union

def check_omi_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    check_antlr_version: bool = True
) -> bool:
    return math_equal(prediction, reference, include_percentage, tolerance, timeout, check_antlr_version)

def check_mathv_equal(
    gold, 
    target, 
    float_rounding: int=6,
    numeric_precision: int=15,
    strict: bool=True,
    timeout_seconds: int=10
) -> bool:
    return verify(gold, target, float_rounding, numeric_precision, strict, timeout_seconds)

def compute_score(data_source, solution_str, ground_truth, extra_info, timeout_seconds:int=10) -> float:
    if "\\boxed" not in solution_str:
        return 0.0
    
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1]
        
    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False
    
    # omi
    try:
        omi_pred = extract_answer(solution_str, extract_from_boxed=True)
        omi_correct = check_omi_equal(omi_pred, ground_truth, check_antlr_version=False, timeout=timeout_seconds)
    except Exception:
        omi_correct = False
    
    # math
    try:
        mathv_pred = parse(solution_str, parsing_timeout=timeout_seconds)
        mathv_correct = check_mathv_equal(parse(str(ground_truth), parsing_timeout=timeout_seconds), mathv_pred, timeout_seconds=timeout_seconds)
    except Exception:
        mathv_correct = False
    
    # deepscaler
    deepscaler_correct = deepscaler_compute_score(data_source, solution_str, ground_truth, extra_info, use_think=True)

    acc = omi_correct or mathv_correct or deepscaler_correct
    score = 1.0 if acc else 0.0
    
    return score

if __name__ == "__main__":
    """
    conda activate zero
    python utils/reward_score/deepmath.py --file_path /mnt/home/al2644/research/projects/perturb-r/results/math8k/benchmark/QwQ-32Btrain.pickle
    """
    import argparse
    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_pickle(args.file_path)
    import pdb; pdb.set_trace()

    tqdm.pandas()
    df["score"] = df.progress_apply(lambda x: compute_score(None, x["pred"], x["gt"], None, 10), axis=1)
    df.to_pickle(args.file_path)