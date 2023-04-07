from ray.rllib.algorithms.sac import SACConfig
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

TASK_NAME = "pick-place-v2"

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment(TASK_NAME))

    return env

register_env(TASK_NAME, env_creator)

class SuccessMetric(DefaultCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies, episode: EpisodeV2, env_index, **kwargs):
        agent_id = list(base_env.get_agent_ids())[0]
        episode.custom_metrics = episode._last_infos[agent_id]


algo = (
    SACConfig()
    .callbacks(SuccessMetric)
    .training(
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        model=ModelConfigDict(
            fcnet_hiddens =  [400, 400],
            fcnet_activation = "relu"
        )
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=4
    )
    .resources(
        num_gpus=0, 
        num_cpus_per_worker=1
    )
    .environment(env = TASK_NAME)
    .build()
)

for i in range(20000):
    result = algo.train()
    print(pretty_print(result))

    if i % 250 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
