from openai_engine import OpenAI_Engine
import pandas as pd
import os
import sys
def r1_distill_math_level3to5(data_dir:str,
                              max_tokens:int=4096,
                              ):
    math_df = pd.read_pickle(os.path.join(data_dir, "input_df.pickle"))
    math_df = math_df[math_df['level'].isin(['Level 3', 'Level 4', 'Level 5'])][:10]

    prompt_template = "Return your final response within \\boxed{{}}. {problem}"
    template_map = {'problem': 'problem'}
    nick_name = "r1_distill_math_level3to5"

    cache_filepath = os.path.join(data_dir, f"responses.pickle")
    engine = OpenAI_Engine(input_df=math_df,
                           prompt_template=prompt_template,
                           template_map=template_map,
                           nick_name=nick_name,
                           cache_filepath=cache_filepath,
                           model="deepseek-reasoner",
                           max_tokens=max_tokens,
                           )
    return engine

if __name__ == "__main__":
    engine = r1_distill_math_level3to5(data_dir="/share/goyal/lio/reasoning/data/hendrycks_math/level3to5/remain")
    engine.run_model(overwrite=False, num_processes=10)

