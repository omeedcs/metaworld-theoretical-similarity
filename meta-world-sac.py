from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gymnasium.wrappers import EnvCompatibility
from MWSamplingEnvironment import MWSamplingEnvironment
import sys

TASK_NAME = "bin-picking-v2"

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment(TASK_NAME))

    return env

register_env(TASK_NAME, env_creator)

algo = (
    SACConfig()
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=0, num_cpus_per_worker=1)
    .environment(env = TASK_NAME)
    .build()
)

for i in range(20000):
    result = algo.train()
    print(pretty_print(result))

    if i % 250 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
