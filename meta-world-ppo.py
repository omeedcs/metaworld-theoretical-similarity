from stable_baselines3 import PPO

import metaworld
import random

ml1 = metaworld.ML1('pick-place-v2')

env = ml1.train_classes['pick-place-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

model = PPO("MlpPolicy", env, verbose=1)
# we need to handle episode termination manually since SB3 doesn't issues #284 and #633
# model.learn(total_timesteps=1000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # seems to render to an offscreen window which we can't see
    # correctly outputs rgb images we can construct into a video, though
    vec_env.render()
    if done:
      obs = env.reset()

env.close()
