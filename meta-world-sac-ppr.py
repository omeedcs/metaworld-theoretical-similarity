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
from stable_baselines3.common import utils
import metaworld
from collections import namedtuple
import imageio
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from enum import Enum
from TimeLimit import TimeLimit
import sys
import time

# # link to wandb.
# wandb.init(
#     project="robin-research-theoretical-similarity",
# )

task_name = sys.argv[1]
past_policy_path = sys.argv[2]
init_reuse_prob = float(sys.argv[3])
reuse_decay = float(sys.argv[4])

# task_name = 'pick-and-place-v1'
curve_png = f'{task_name}/training_curve.png'
video_dir = f'{task_name}/training_mp4s' 
model_dir = f'{task_name}/models'

input(f"about to delete {task_name}/. Ready?")
os.system(f"rm -r {task_name}")
os.mkdir(task_name)

ml1 = metaworld.ML1(task_name)

env = ml1.train_classes[task_name]()
task = ml1.train_tasks[0]
env.set_task(task)
env = TimeLimit(env, 500)

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
    'train_freq': (100, 'step'),
    'device': 'cpu',
}

model = SAC("MlpPolicy", env, verbose = 1, **params)

# create past policy model and load parameters
past_model = SAC("MlpPolicy", env, verbose = 1, **params)
infile = open(past_policy_path, "rb")
loaded_params = pickle.load(infile)
infile.close()

past_model.set_parameters(loaded_params)

class EmptyCallback(BaseCallback):
    """
    doesn't do anything anymore
    """
    def __init__(self, verbose = 0):
        super(EmptyCallback, self).__init__(verbose)
        self.path_length = 0

    def _on_training_start(self):
        pass

    def _on_rollout_start(self):
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
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

def collect_ppr_rollouts(
    self,
    past_model,
    reuse_prob,
    env,
    callback,
    train_freq,
    replay_buffer,
    action_noise = None,
    learning_starts = 0,
    log_interval = None,
):
    from gym import spaces
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
    from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
    """
    Collect experiences and store them into a ``ReplayBuffer``.
    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param train_freq: How much experience to collect
        by doing rollouts of current policy.
        Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
        or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
        with ``<n>`` being an integer greater than 0.
    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param replay_buffer:
    :param log_interval: Log data every ``log_interval`` episodes
    :return:
    """
    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    assert isinstance(env, VecEnv), "You must pass a VecEnv"
    assert train_freq.frequency > 0, "Should at least collect one step or episode."

    if env.num_envs > 1:
        assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

    # Vectorize action noise if needed
    if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
        action_noise = VectorizedActionNoise(action_noise, env.num_envs)

    if self.use_sde:
        self.actor.reset_noise(env.num_envs)

    callback.on_rollout_start()
    continue_training = True

    while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.actor.reset_noise(env.num_envs)

        # Sample action from learning policy or from past policy
        actions, buffer_actions = None, None
        if random.random() < reuse_prob:
            # take the action the past model would do
            unscaled_action, _ = past_model.predict(self._last_obs, deterministic=True)

            # Rescale the action from [low, high] to [-1, 1]
            if isinstance(self.action_space, spaces.Box):
                scaled_action = self.policy.scale_action(unscaled_action)

                # Add noise to the action (improve exploration)
                if action_noise is not None:
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

                # We store the scaled action in the buffer
                buffer_actions = scaled_action
                actions = self.policy.unscale_action(scaled_action)
            else:
                # Discrete case, no need to normalize or clip
                buffer_actions = unscaled_action
                actions = buffer_actions
        else:
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

        # Rescale and perform action
        new_obs, rewards, dones, infos = env.step(actions)

        self.num_timesteps += env.num_envs
        num_collected_steps += 1

        # Give access to local variables
        callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1
                self._episode_num += 1

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
    callback.on_rollout_end()

    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

x_time_steps = []
y_total_reward = []
y_success_rate = []

total_timesteps = 3_000_000
total_timesteps, callback = model._setup_learn(total_timesteps, callback = EmptyCallback())
reuse_prob = init_reuse_prob

while model.num_timesteps < total_timesteps:
    start_time = time.time()
    # periodically evaluate
    if model.num_timesteps % 10000 == 0:
        total_reward, success_rate = evaluate_model(model)
        model.save(f"{model_dir}/{task_name}_{model.num_timesteps}.pkl")
        x_time_steps.append(model.num_timesteps)
        y_total_reward.append(total_reward[0])
        y_success_rate.append(rgb2hex((1.0 - success_rate, success_rate, 0.0)))
        print(f"({model.num_timesteps} timesteps): {total_reward}")

        plt.scatter(x_time_steps, y_total_reward, c=y_success_rate)
        plt.xlabel("Timesteps Trained")
        plt.ylabel("Total Reward per Episode")
        plt.savefig(curve_png)
        # wandb.log({
        #     "Timesteps Trained": x_time_steps[-1], 
        #     "Total Reward per Episode": y_total_reward[-1]
        #         })
        #     
    model.get_env().reset()

    # collect experience using PPR
    collect_ppr_rollouts(model, past_model, reuse_prob, model.env, callback, model.train_freq, replay_buffer = model.replay_buffer)

    if model.num_timesteps > model.learning_starts:
        model.train(gradient_steps = 250)
        if model.num_timesteps % 10000 == 0:
            render_model(model, file_name=task_name+"-s{:07d}-r{}".format(model.num_timesteps, total_reward[0]))

    print(f"taken {time.time() - start_time} seconds")


x_time_steps.append(model.num_timesteps)
total_reward = evaluate_model(model)[0]
y_total_reward.append(total_reward)
print(f"({model.num_timesteps} timesteps): {total_reward}")
render_model(model, file_name=task_name+"-s{:07d}-r{}".format(model.num_timesteps, total_reward))

env.close()
