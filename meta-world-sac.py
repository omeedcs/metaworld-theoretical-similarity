from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gymnasium.wrappers import EnvCompatibility
from MWSamplingEnvironment import MWSamplingEnvironment

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment("reach-v2"))
    return env

register_env("mw_sampling_env", env_creator)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env = "mw_sampling_env", disable_env_checking=True)
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
