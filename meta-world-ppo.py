from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import metaworld
import random

ml1 = metaworld.ML1('pick-place-v2')

env = ml1.train_classes['pick-place-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

model = PPO("MlpPolicy", env, verbose=1)
# we need to handle episode termination manually since SB3 doesn't issues #284 and #633
# model.learn(total_timesteps=1000)

class MWTerminationCallback(BaseCallback):
    """
    A custom callback that terminates the rollout procedure early when we've reached the maximum
    steps of the metaworld environment.
    """

    def __init__(self, verbose = 0):
        super(MWTerminationCallback, self).__init__(verbose)
        self.step_count = 0

    def _on_training_start(self):
        pass

    def _on_rollout_start(self):
        pass

    def _on_step(self) -> bool:
        """
        Returns false if we've reached the the maximimum number of steps for this MetaWorld environment
        """
        self.step_count += 1
        if self.step_count > 500:
            return False
        return True

    def _on_rollout_end(self):
        pass

    def _on_training_end(self):
        pass

# manual learning
total_timesteps = 10_000
iteration = 0
total_timesteps, callback = model._setup_learn(total_timesteps, callback=MWTerminationCallback())
while model.num_timesteps < total_timesteps:
    continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    if not continue_training:
        break

    iteration += 1
    model.train()

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # seems to render to an offscreen window which we can't see
    # correctly outputs rgb images we can construct into a video, though
    vec_env.render()
    if done:
      obs = env.reset()

env.close()
