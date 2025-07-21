import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.openmathinst_utils import extract_answer, math_equal

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
    if "</think>" in solution_str and "\\boxed" in solution_str:
        # to avoid the case that boxed appears in both the thinking and the solution
        solution_str = solution_str.split("</think>")[-1]
    else:
        return 0.0
        
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
    
    acc = omi_correct or mathv_correct
    score = 1.0 if acc else 0.0
    
    return score