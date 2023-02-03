from typing import Callable
import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import metaworld
import imageio
import numpy as np

task_name = 'pick-place-v2'

ml1 = metaworld.ML1(task_name)

env = ml1.train_classes[task_name]()
task = ml1.train_tasks[0]
env.set_task(task)

params = {
    # "learning_rate": 5e-4,
    "gamma": 0.99,
}

model = PPO("MlpPolicy", env, verbose=1, **params)
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

def evaluate_model(model, T=10):
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
        img = vec_env.render(mode="rgb_array")

    imageio.mimsave(f"{file_name}.gif", [np.array(img) for i, img in enumerate(images) if i % 5 == 0])


# manual learning
total_timesteps = 1_000_000
iteration = 0
total_timesteps, callback = model._setup_learn(total_timesteps, callback=MWTerminationCallback())
while model.num_timesteps < total_timesteps:
    # periodically evaluate
    if iteration % 10 == 0:
        print(f"Iteration {iteration} ({model.num_timesteps} timesteps): {evaluate_model(model)}")

    if iteration % 100 == 0:
        print("Creating render")
        render_model(model)

    model.get_env().reset()
    continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    iteration += 1

    model.train()


print(f"Completed {iteration} iterations")

print(evaluate_model(model))
render_model(model)

env.close()
