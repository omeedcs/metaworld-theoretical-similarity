import gym
import metaworld
import numpy as np
import torch
import random
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


# To make the evn can be applied to the SAC. SAC is picky for the gym type env.
class CustomEnv(gym.Env):
    def __init__(self, metaworld_env):
        super().__init__()
        self.env = metaworld_env
        self.observation_space = metaworld_env.observation_space
        self.action_space = metaworld_env.action_space

    def reset(self):
        obs = self.env.reset()
        # print("The obs type is: ", obs.dtype)
        obs = obs.astype(np.float32)
        # print("The obs is: ", obs)
        return obs

    def step(self, action):
        action = action.astype(np.float32)
        obs, reward, done, info = self.env.step(action)
        # print("The obs type after step is: ", obs.dtype)
        obs = obs.astype(np.float32)
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()


task_name = 'pick-place-v2'

ml1 = metaworld.ML1(task_name)
env = ml1.train_classes[task_name]()
task = random.choice(ml1.train_tasks)
env.set_task(task)

env = CustomEnv(env)


print(check_env(env))


