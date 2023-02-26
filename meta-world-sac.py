# REFERENCE LINK: 
# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
# https://www.reddit.com/r/reinforcementlearning/comments/wztujn/agent_trains_great_with_ppo_but_terrible_with_sac/


# import wandb
from typing import Callable
import random
import os
import math
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import metaworld
from collections import namedtuple
import imageio
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from enum import Enum

# # link to wandb.
# wandb.init(
#     project="robin-research-theoretical-similarity",
# )

task_name = 'pick-place-v2'
# task_name = 'pick-and-place-v1'
video_dir = 'training_mp4s' 

os.system(f"rm -r {video_dir}")

ml1 = metaworld.ML1(task_name)

env = ml1.train_classes[task_name]()
task = ml1.train_tasks[0]
env.set_task(task)

policy_kwargs = dict(activation_fn= th.nn.ReLU,
                     net_arch=[256, 256])

params = {
    "batch_size": 500,
    'learning_rate': 3e-4,
    # exploration strategy will remain as default.
    "use_sde": False,
    # 'replay_buffer_kwargs': replay_kwargs,
    'policy_kwargs': policy_kwargs,
    'tau': 5e-3,
    'buffer_size': 1000000,
    'train_freq': (10, 'step'),
}

# log_std_init = math.exp(-20.0)? 

model = SAC("MlpPolicy", env, verbose = 1, **params)
# we need to handle episode termination manually since SB3 doesn't issues #284 and #633
# model.learn(total_timesteps=1000)

class MWTerminationCallback(BaseCallback):
    """
    A custom callback that terminates the rollout procedure early when we've reached the maximum
    steps of the metaworld environment.
    """

    def __init__(self, verbose = 0):
        super(MWTerminationCallback, self).__init__(verbose)
        self.path_length = 0

    def _on_training_start(self):
        pass

    def _on_rollout_start(self):
        pass

    def _on_step(self) -> bool:
        # TODO: Check this for safety such that returns are not
        # estimated across multiple episodes via discount factor
        self.path_length += 1

        if self.path_length >= 500:
            self.model.env.reset()
            self.path_length = 0
        return True

    def _on_rollout_end(self):
        """
        Manually reset environment since MW won't do it itself
        """
        self.model.env.reset()
        pass

    def _on_training_end(self):
        pass



def evaluate_model(model, T=2):
    """
    Runs the model over T total evaluations and gets the average
    total return
    """
    rewards = 0
    for i in range(T):
        vec_env = model.get_env()
        obs = vec_env.reset()
        N = 500
        for i in range(N):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            # print the current reward.

            print(f"Current Reward: {reward}")
            # obtain success from info
            success = info[0]
            print(f"Success: {success}")

            rewards += reward
            # seems to render to an offscreen window which we can't see
            # correctly outputs rgb images we can construct into a video, though
            # vec_env.render()

    return rewards / T

def render_model(model, file_name=task_name):
    """
    Runs a simulation of the task and renders a view of it to a gif
    in the current working directory.
    """
    print("Creating render")
    images = []
    vec_env = model.get_env()
    img = vec_env.render(mode="rgb_array")
    obs = vec_env.reset()
    N = 500
    for i in range(N):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # seems to render to an offscreen window which we can't see
        # correctly outputs rgb images we can construct into a video, though
        img = vec_env.render(mode = "rgb_array")

    if not os.path.isdir(f"{video_dir}"):
        os.mkdir(video_dir)
    path = os.path.join(video_dir, f"{file_name}")
    imageio.mimsave(path+".gif", [np.array(img) for i, img in enumerate(images) if i % 5 == 0])
    os.system(f'ffmpeg -i {path}.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {path}.mp4')
    os.system(f'rm {path}.gif')

x_time_steps = []
y_total_reward = []

# manual learning
# at what point do we overfit?
total_timesteps = 10_000_000
iteration = 0
total_timesteps, callback = model._setup_learn(total_timesteps, callback = MWTerminationCallback())

while model.num_timesteps < total_timesteps:
    # periodically evaluate
    if iteration % 25 == 0:
        total_reward = evaluate_model(model)[0]
        x_time_steps.append(model.num_timesteps)
        y_total_reward.append(total_reward)
        print(f"Iteration {iteration} ({model.num_timesteps} timesteps): {total_reward}")
        render_model(model, file_name=task_name+"-s{:07d}-r{}".format(model.num_timesteps, total_reward))

        plt.plot(x_time_steps, y_total_reward)
        plt.xlabel("Timesteps Trained")
        plt.ylabel("Total Reward per Episode")
        plt.savefig("training_curve.png")
        # wandb.log({
        #     "Timesteps Trained": x_time_steps[-1], 
        #     "Total Reward per Episode": y_total_reward[-1]
        #         })

    model.get_env().reset()
    continue_training = model.collect_rollouts(model.env, callback, model.train_freq, replay_buffer = model.replay_buffer)

    iteration += 1

    model.train(gradient_steps = 500)


x_time_steps.append(model.num_timesteps)
total_reward = evaluate_model(model)[0]
y_total_reward.append(total_reward)
print(f"Iteration {iteration} ({model.num_timesteps} timesteps): {total_reward}")
render_model(model, file_name=task_name+"-s{:07d}-r{}".format(model.num_timesteps, total_reward))

env.close()