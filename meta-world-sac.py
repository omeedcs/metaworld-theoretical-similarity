# REFERENCE LINKS: 
# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
# https://www.reddit.com/r/reinforcementlearning/comments/wztujn/agent_trains_great_with_ppo_but_terrible_with_sac/
# we need to handle episode termination manually since SB3 doesn't issues #284 and #633


# import wandb
from typing import Callable
import random
import os
import math
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from collections import namedtuple
import imageio
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from enum import Enum
from TimeLimit import TimeLimit
import sys
import time
import wandb

# link to wandb.
wandb.init(
    project="robin-research-theoretical-similarity",
)

task_name = sys.argv[1]
# task_name = 'pick-and-place-v1'
curve_png = f'{task_name}/training_curve.png'
video_dir = f'{task_name}/training_mp4s' 
model_dir = f'{task_name}/models'

env_constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{task_name}-goal-observable']

os.system(f"rm -r {task_name}")
os.mkdir(task_name)
os.mkdir(model_dir)

def sample_env(env_constructor):
    while True:
        new_env = env_constructor(seed=random.randint(0, 10000))
        new_env.reset()
        yield TimeLimit(new_env, 500)

env_sampler = sample_env(env_constructor)
env = next(env_sampler)

policy_kwargs = dict(activation_fn= th.nn.ReLU,
                     net_arch=[256, 256])

params = {
    "batch_size": 500,
    'learning_rate': 3e-4,
    "use_sde": False,
    'policy_kwargs': policy_kwargs,
    'tau': 5e-3,
    'buffer_size': 100000,
    # has massive impact on training speed.
    'train_freq': (500, 'step'),
    'device': 'cpu',
}

model = SAC("MlpPolicy", env, verbose = 1, **params)
new_env = next(env_sampler)
# TODO: why does this work? force_reset was supposed to *avoid* errors
model.set_env(new_env, force_reset=False)

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
        return True

    def _on_rollout_end(self):
        """
        Reset the environment and sample new env variation
        """
        global env_sampler
        new_env = next(env_sampler)
        self.model.set_env(new_env, force_reset = False)
        pass

    def _on_training_end(self):
        pass

def evaluate_model(model, T=2):
    """
    Runs the model over T total evaluations and gets the average
    total return
    """
    rewards = 0
    successes = 0
    for i in range(T):
        vec_env = model.get_env()
        obs = vec_env.reset()
        N = 500
        any_success = False
        for i in range(N):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            success = info[0]
            if success['success'] > 0: any_success = True
            rewards += reward
        if any_success: successes += 1.0

    return rewards / T, successes / T

def verified_evaluation(model, T=2):
    """
    Evaluates, saves parameters, and revaluates to verify that the model
    was saved correctly
    """
    import pickle

    num_timesteps = model.num_timesteps
    reward, successes = evaluate_model(model, T=T)

    # try saving and reloading the model for verification
    to_save = model.get_parameters()
    outfile_name = f"{task_name}/models/{task_name}-s{num_timesteps}-r{reward.item()}.pkl"
    outfile = open(outfile_name, "wb")
    pickle.dump(to_save, outfile)
    outfile.close()

    infile = open(outfile_name, "rb")
    loaded_params = pickle.load(infile)
    infile.close()

    model.set_parameters(loaded_params)

    loaded_reward, loaded_successes = evaluate_model(model, T=T)
    assert(reward == loaded_reward)

    return loaded_reward, loaded_successes

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
        img = vec_env.render(mode = "rgb_array")

    if not os.path.isdir(f"{video_dir}"):
        os.makedirs(video_dir, exist_ok=True)
    path = os.path.join(video_dir, f"{file_name}")
    imageio.mimsave(path+".gif", [np.array(img) for i, img in enumerate(images) if i % 5 == 0])
    os.system(f'ffmpeg -i {path}.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {path}.mp4'  + ' > /dev/null 2>&1')
    os.system(f'rm {path}.gif')

x_time_steps = []
y_total_reward = []
y_success_rate = []

total_timesteps = 3_000_000
iteration = 0
total_timesteps, callback = model._setup_learn(total_timesteps, callback = MWTerminationCallback())

while model.num_timesteps < total_timesteps:
    start_time = time.time()
    # periodically evaluate
    if iteration % 100 == 0:
        total_reward, success_rate = verified_evaluation(model)
        x_time_steps.append(model.num_timesteps)
        y_total_reward.append(total_reward[0])
        y_success_rate.append(rgb2hex((1.0 - success_rate, success_rate, 0.0)))
        print(f"Iteration {iteration} ({model.num_timesteps} timesteps): {total_reward}")

        plt.scatter(x_time_steps, y_total_reward, c=y_success_rate)
        plt.xlabel("Timesteps Trained")
        plt.ylabel("Total Reward per Episode")
        plt.savefig(curve_png)
        wandb.log({
            "Timesteps Trained": x_time_steps[-1], 
            "Total Reward per Episode": y_total_reward[-1]
                })
            
    model.get_env().reset()
    continue_training = model.collect_rollouts(model.env, callback, model.train_freq, replay_buffer = model.replay_buffer)


    if model.num_timesteps > model.learning_starts:
        model.train(gradient_steps = 500)
        if iteration % 100 == 0:
            render_model(model, file_name=task_name+"-s{:07d}-r{}".format(model.num_timesteps, total_reward[0]))

    print(f"Iteration {iteration} taken {time.time() - start_time} seconds")
    iteration += 1


x_time_steps.append(model.num_timesteps)
total_reward = evaluate_model(model)[0]
y_total_reward.append(total_reward)
print(f"Iteration {iteration} ({model.num_timesteps} timesteps): {total_reward}")
render_model(model, file_name=task_name+"-s{:07d}-r{}".format(model.num_timesteps, total_reward))

wrapped_env.close()

