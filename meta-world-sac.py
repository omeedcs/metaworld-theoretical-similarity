from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gymnasium.wrappers import EnvCompatibility
from MWSamplingEnvironment import MWSamplingEnvironment
import sys

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment("soccer-v2"))

    return env

register_env("soccer-v2", env_creator)

algo = (
    SACConfig()
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=0)
    .environment(env = "soccer-v2")
    .build()
)

for i in range(20000):
    result = algo.train()
    print(pretty_print(result))

    if i % 250 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
