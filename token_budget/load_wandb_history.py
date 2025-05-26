import pandas as pd
import wandb
import os
import wandb

wandb.login()
api = wandb.Api()

"""
NOTE: run id is not run name. To get run id

for r in api.runs(f"{ENTITY}/{PROJECT}"):
    pprint({"id": r.id, "name": r.name})
"""

ENTITY = "al2644-cornell-university"
PROJECT = "sky-math-8k"
RUN_ID = "8lsf1hr8" 
RUN_DIR = "/share/goyal/lio/reasoning/model/sky_math8k/sft/Qwen2.5_Math_1.5B_sky_math8k_max_length_4096_bsz_32_epochs_10"

run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

history = run.history()
checkpoints = [int(fname.replace("checkpoint-", ""))for fname in os.listdir(RUN_DIR) if "checkpoint-" in fname]

df = []
for checkpoint in checkpoints:
    row = history[history['train/global_step'] <= checkpoint].dropna().iloc[-1:]
    row['checkpoint'] = checkpoint
    df.append(row)
df = pd.concat(df, ignore_index = True)

df.to_pickle(f"{RUN_DIR}/wandb_history.pickle")
