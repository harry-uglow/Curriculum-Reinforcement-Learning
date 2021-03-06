import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from envs.ImageObsVecEnvWrapper import SimImageObsVecEnvWrapper
from envs.ResidualVecEnvWrapper import ResidualVecEnvWrapper
from envs.wrappers import PoseEstimatorVecEnvWrapper, InitialController, BoundPositionVelocity, \
    ScaleActions
from a2c_ppo_acktr.tuple_tensor import TupleTensor

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

exploration_factor = 1/3


def make_env(env_name, scene_path, seed, rank, log_dir, allow_early_resets, vis, init_control):
    def _thunk():
        env = env_name(scene_path, rank, not vis)

        env.seed(seed + rank)

        env = BoundPositionVelocity(env)
        if init_control:
            env = InitialController(env)
        env = ScaleActions(env, 0.05 * exploration_factor)  # 1 step = 0.05 ms

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        return env

    return _thunk


def wrap_initial_policies(envs, device, initial_policies):
    if initial_policies:
        curr_ip, ob_rms, more_ips = initial_policies
        envs = wrap_initial_policies(envs, device, more_ips)
        return ResidualVecEnvWrapper(envs, curr_ip, ob_rms, device)
    return envs


def make_vec_envs(env_name, scene_path, seed, num_processes, gamma, log_dir, device,
                  allow_early_resets, initial_policies, num_frame_stack=None, show=False,
                  no_norm=False, pose_estimator=None, image_ips=None, init_control=True):
    envs = [make_env(env_name, scene_path, seed, i, log_dir, allow_early_resets, show, init_control)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = wrap_initial_policies(envs, device, initial_policies)

    if pose_estimator is not None:
        envs = SimImageObsVecEnvWrapper(envs)

    if len(envs.observation_space.shape) == 1 and not no_norm:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if pose_estimator is not None:
        envs = wrap_initial_policies(envs, device, image_ips)
        envs = PoseEstimatorVecEnvWrapper(envs, device, *pose_estimator, abs_to_rel=True)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif not pose_estimator and len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                obs_shape[self.op[0]],
                obs_shape[self.op[1]],
                obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, tuple):
            obs = TupleTensor(torch.from_numpy(obs[0]).float().to(self.device),
                              torch.from_numpy(obs[1]).float().to(self.device))
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, tuple):
            obs = TupleTensor(torch.from_numpy(obs[0]).float().to(self.device),
                              torch.from_numpy(obs[1]).float().to(self.device))
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
