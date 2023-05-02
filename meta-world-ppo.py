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

import sys
from typing import Union

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

TASK_NAME = "soccer-v2"

writer = SummaryWriter(f"./runs/{TASK_NAME}")

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment(TASK_NAME))
    return env

register_env(TASK_NAME, env_creator)

class SuccessMetric(DefaultCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies, episode: EpisodeV2, env_index, **kwargs):
        agent_id = list(base_env.get_agent_ids())[0]
        episode.custom_metrics = episode._last_infos[agent_id]


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
    .environment(env = TASK_NAME)
    .build()
)

num_episodes = 1000
success_episodes = 0

for i in range(num_episodes):
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
