# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# rm -rf verl/trainer/ppo/ray_trainer.py; vim verl/trainer/ppo/ray_trainer.py
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, dataprotoitem_to_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.checkpoint.push import push_model_to_hf

import tempfile
from filelock import FileLock
import json
from collections import Counter
import wandb
import re
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from datasets import Dataset

from verl.trainer.ppo.ray_trainer import *

WorkerType = Type[Worker]

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

class RayDAPOTrainer(RayPPOTrainer):
    def fit_collect(self):
        """
        The training loop of DAPO with reasoning trace collection.
        The driver process only need to call the compute functions of the worker group through RPC to construct the DAPO dataflow.
        The light-weight advantage computation is done on the driver process.

        Here are the main differences from the PPO trainer:
        1. Clip Higher: enable clip_high and clip_low in the actor update
        2. Overlong Buffer: enable overlong buffer in RewardManager
        3. Token-level Policy Gradient Loss: enable token-level policy gradient loss in the actor update
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          wandb_run_id=self.config.trainer.wandb_run_id)

        self.global_steps = 0
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True) and not self.config.trainer.get('skip_val', False):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        # we start from step 1
        # HACK: Record total tokens
        self.global_steps += 1
        self.token_budget = self.config.trainer.token_budget
        self.load_rollout_dataset()
        self.load_global_total_tokens()

        total_training_steps = self.config.trainer.total_epochs * len(self.train_dataloader)
        # If resuming, start tqdm at the correct position
        global_step_tqdm = tqdm(
            range(total_training_steps),
            desc="Global Training Steps",
            unit="step",
            total=total_training_steps,
            leave=True,
            initial=self.global_steps
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                score_records = []

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # HACK: Record total tokens
                    responses_length = batch.batch['responses'].size(-1)
                    self.global_total_tokens += batch.batch['attention_mask'][:, -responses_length:].sum().item()
                    metrics.update({'token_budget/total_token_budget': self.global_total_tokens})
                    # End of HACK

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        reward_tensor, score_record = self.reward_fn(batch)
                        score_records.extend(score_record)
                        batch.batch['token_level_scores'] = reward_tensor

                        #HACK: Update reasoning dataset
                        self.update_rollout_dataset(batch, reward_tensor, score_record)
                        # END OF HACK
                        
                        # HACK: s
                        # 1. Record # solve none and solve all and avg solve rate
                        # 2. remove sequences that either solve none or solve all
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0

                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid (w soft length penalty it can be <0 or <1)
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards > 0).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                        
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all
                        metrics['batch/avg_solve_rate'] = batch.batch['token_level_scores'].sum(-1).mean().item()
                        
                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]
                            batch = dataprotoitem_to_dataproto(batch)
                            # Round down to the nearest multiple of world size
                            num_trainer_replicas = self.actor_rollout_wg.world_size 
                            max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                            if not max_batch_size:
                                # give up, you got everything either all wrong or right.
                                continue

                            size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                            size_mask[:max_batch_size] = True
                            batch = batch[size_mask]
                            batch = dataprotoitem_to_dataproto(batch)
                        # End of HACK
                        # compute rewards. apply_kl_penalty if available

                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False) and self.config.actor_rollout_ref.actor.get('kl_loss_coef', 0.0) > 0.0:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    with _timer('log_scores', timing_raw):
                        score_metrics = self._log_scores_to_wandb(score_records, epoch)
                        # Update metrics with returned values instead of direct logging
                        # logger.log(data=score_metrics, step=self.global_steps)
                        metrics.update(score_metrics)

                    # validate + push rollout dataset to huggingface
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        
                        if not self.config.trainer.get('skip_val', False):
                            with _timer('testing', timing_raw):
                                val_metrics: dict = self._validate()
                            metrics.update(val_metrics)

                        # FIXED: Add error handling for saving rollout dataset
                        try:
                            success = self.save_rollout_dataset()
                            if success:
                                print(f"Successfully saved rollout dataset at step {self.global_steps}")
                            else:
                                print(f"Failed to save rollout dataset at step {self.global_steps}")
                        except Exception as e:
                            print(f"Error saving rollout dataset at step {self.global_steps}: {e}")

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        self._save_history_accuracy(self.history_accuracy)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                global_step_tqdm.update(1)

                if self.global_steps >= self.total_training_steps or \
                    (self.token_budget is not None and self.global_total_tokens >= self.token_budget):
                    # perform validation after training
                    if self.val_reward_fn is not None and not self.config.trainer.get('skip_val', False):
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    
                    # save final rollout dataset after training
                    try:
                        self.save_rollout_dataset()
                        print("Final rollout dataset saved successfully")
                    except Exception as e:
                        print(f"Error saving final rollout dataset: {e}")
                    
                    # save the final checkpoint
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return