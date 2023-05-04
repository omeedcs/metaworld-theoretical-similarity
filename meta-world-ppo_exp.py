from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog, ModelConfigDict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.env.base_env import BaseEnv
from gymnasium.wrappers import EnvCompatibility
from MWSamplingEnvironment import MWSamplingEnvironment

from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import Policy
from typing import Dict

import sys
from typing import Union
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import json
import os
import numpy as np

TASK_NAME = "basketball-v2"

writer = SummaryWriter(f"./runs/{TASK_NAME}")

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment(TASK_NAME))
    return env

register_env(TASK_NAME, env_creator)

class SuccessMetric(DefaultCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies, episode: EpisodeV2, env_index, **kwargs):
        agent_id = list(base_env.get_agent_ids())[0]
        episode.custom_metrics = episode._last_infos[agent_id]

    def on_postprocess_trajectory(
        self, *, worker: RolloutWorker, episode: MultiAgentEpisode, agent_id: str,
        policy_id: str, policies: Dict[str, Policy], postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch], **kwargs):
        # Get the agent's last observation, action, reward, done flag, and info from the postprocessed batch
        last_observation = postprocessed_batch[SampleBatch.CUR_OBS]
        last_action = postprocessed_batch[SampleBatch.ACTIONS]
        last_reward = postprocessed_batch[SampleBatch.REWARDS]
        last_done = postprocessed_batch[SampleBatch.DONES]
        last_info = postprocessed_batch[SampleBatch.INFOS]

         # Convert NumPy arrays in last_info to lists
        last_info_list = []
        for info in last_info:
            converted_info = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    converted_info[key] = value.tolist()
                else:
                    converted_info[key] = value
            last_info_list.append(converted_info)
        # Store these in the episode's custom metrics
        temp={}
        temp['last_observation'] = last_observation.tolist()
        temp['last_action'] = last_action.tolist()
        temp['last_reward'] = last_reward.tolist()
        temp['last_done'] = last_done.tolist()
        temp['last_info'] = last_info_list
        
        # Create the directory if it doesn't exist
        os.makedirs(f'./exp_collect/{TASK_NAME}', exist_ok=True)
        # Append the experiences to the JSON file
        with open(f'./exp_collect/{TASK_NAME}/{TASK_NAME}_exp.json', 'a') as f:
            f.write(json.dumps(temp) + "\n")

algo = (
    PPOConfig()
    .callbacks(SuccessMetric)
    .training(
        model=ModelConfigDict(
            fcnet_hiddens =  [512, 512],
            fcnet_activation = "tanh"
        )
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=1
    )
    .resources(
        num_gpus=1, 
        num_gpus_per_worker=1
    )
    .environment(env = TASK_NAME,
                 render_env = True)
    .build()
)

iterations = 20000
success_episodes = 0

for i in range(iterations):
    result = algo.train()
    # print(pretty_print(result))
    # print(pretty_print(result['custom_metrics']))
    if(result['custom_metrics']['success_mean'] > 0):
        success_episodes += 1
        print("The success times is: ", success_episodes)

    writer.add_scalar("episode_reward_mean/train", result['episode_reward_mean'], i)
    success_rate = round(success_episodes/float(i+1.0), 5)
    writer.add_scalar("success rate/train", success_rate, i)

    if i % 250 == 0:
        checkpoint_dir = algo.save(f"./Checkpoints/{TASK_NAME}/")
        print(f"Checkpoint saved in directory {checkpoint_dir}")

# save the policy
policy = algo.get_policy()
torch.save(policy.model.state_dict(), f"./saved_policies/{TASK_NAME}_policy.pth")

writer.flush()
writer.close()
