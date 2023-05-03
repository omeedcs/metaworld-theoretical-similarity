import os
import glob
import imageio
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
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


TASK_NAME = "soccer-v2"

def env_creator(env_config):
    env = EnvCompatibility(MWSamplingEnvironment(TASK_NAME))
    return env

register_env(TASK_NAME, env_creator)

class SuccessMetric(DefaultCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies, episode: EpisodeV2, env_index, **kwargs):
        agent_id = list(base_env.get_agent_ids())[0]
        episode.custom_metrics = episode._last_infos[agent_id]

new_algo = (
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
                 render_env = True )
    .build()
)

new_algo.restore("./Checkpoints/soccer-v2/checkpoint_000751")
new_policy = new_algo.get_policy()
# new_policy.model.load_state_dict(torch.load(f"./saved_policies/{TASK_NAME}_policy.pth"))

num_episodes = 100
success_episodes = 0
env = env_creator({})
success_rewards = []

# Save images to a temporary directory
temp_dir = "./tmp/render"
os.makedirs(temp_dir, exist_ok=True)

for i in range(num_episodes):
    # Create the environment again

    obs = env.reset()
    if(type(obs) is tuple):
        obs = obs[0]
    
    done = False
    steps = 0
    total_reward = 0.0
    while(not done):
        steps += 1
        action = new_policy.compute_single_action(obs)[0]
        obs, reward, done, _, info = env.step(action)

        img = env.render() # mode='rgb_array'
        imageio.imwrite(os.path.join(temp_dir, f"{steps}.png"), img)

        total_reward += reward
        if(done or info['success']>0):
            if(info['success']>0):
                success_episodes += 1
                success_rewards.append(total_reward)
            print(i+1, steps, "This episode is successful:", info['success'], "The total reward is: ", total_reward)
            break
    
    # Convert images to gif
    images = []
    for filename in sorted(glob.glob(os.path.join(temp_dir, "*.png"))):
        images.append(imageio.imread(filename))
    imageio.mimsave(f"./tmp/videos/{i+1}_movie.gif", images)

    # Clean up the temporary directory
    for filename in glob.glob(os.path.join(temp_dir, "*.png")):
        os.remove(filename)

display.stop()
#print("The success rate is: ", round(success_episodes/float(num_episodes), 4))
print("The success time is: ", success_episodes)
print("All the success rewards: ", success_rewards)


