import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import *

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# manual learning
total_timesteps = 10_000
iteration = 0
total_timesteps, callback = model._setup_learn(total_timesteps)
while model.num_timesteps < total_timesteps:
    continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    if not continue_training:
        break

    iteration += 1
    model.train()

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

