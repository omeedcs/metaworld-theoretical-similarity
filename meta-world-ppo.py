from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import metaworld
import random

ml1 = metaworld.ML1('pick-place-v2')

env = ml1.train_classes['pick-place-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

model = PPO("MlpPolicy", env, verbose=1, n_steps=500, batch_size=50)
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
        return True

    def _on_rollout_end(self):
        """
        Manually reset environment since MW won't do it itself
        """
        self.model.env.reset()
        pass

    def _on_training_end(self):
        pass

def evaluate_model(model):
    vec_env = model.get_env()
    obs = vec_env.reset()
    N = 500
    rewards = 0
    for i in range(N):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        rewards += reward
        # seems to render to an offscreen window which we can't see
        # correctly outputs rgb images we can construct into a video, though
        vec_env.render()

    env.close()
    return rewards

# manual learning
total_timesteps = 1_000_000
iteration = 0
total_timesteps, callback = model._setup_learn(total_timesteps, callback=MWTerminationCallback())
while model.num_timesteps < total_timesteps:
    model.get_env().reset()
    continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

    iteration += 1
    
    # periodically evaluate
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: {evaluate_model(model)}")

    model.train()

print(f"Completed {iteration} iterations")
