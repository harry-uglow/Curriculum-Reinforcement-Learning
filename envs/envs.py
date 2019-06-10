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

from envs.DRNoWaypointEnv import DRNonRespondableEnv
from envs.DRWaypointEnv import DRWaypointEnv
from envs.DRSparseEnv import DRSparseEnv
from envs.ResidualVecEnvWrapper import ResidualVecEnvWrapper
from envs.wrappers import ImageObsVecEnvWrapper, PoseEstimatorVecEnvWrapper, \
    ClipActions, E2EVecEnvWrapper
from a2c_ppo_acktr.tuple_tensor import TupleTensor

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

# try:
#     import pybullet_envs
# except ImportError:
#     pass


def make_env(scene_path, seed, rank, log_dir, add_timestep, allow_early_resets, vis):
    def _thunk():
        env = DRSparseEnv(scene_path, rank, not vis)

        env.seed(seed + rank)

        env = ClipActions(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        return env

    return _thunk

    # def _thunk():
    #     if env_id.startswith("dm"):
    #         _, domain, task = env_id.split('.')
    #         env = dm_control2gym.make(domain_name=domain, task_name=task)
    #     else:
    #         env = gym.make(env_id)
    #
    #     is_atari = hasattr(gym.envs, 'atari') and isinstance(
    #         env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    #     if is_atari:
    #         env = make_atari(env_id)
    #
    #     env.seed(seed + rank)
    #
    #     obs_shape = env.observation_space.shape
    #
    #     if add_timestep and len(
    #             obs_shape) == 1 and str(env).find('TimeLimit') > -1:
    #         env = AddTimestep(env)
    #
    #     if log_dir is not None:
    #         env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
    #                             allow_early_resets=allow_early_resets)
    #
    #     if is_atari:
    #         if len(env.observation_space.shape) == 3:
    #             env = wrap_deepmind(env)
    #     elif len(env.observation_space.shape) == 3:
    #         raise NotImplementedError("PoseEstimator models work only for atari,\n"
    #             "please use a custom wrapper for a custom pixel input env.\n"
    #             "See wrap_deepmind for an example.")
    #
    #     # If the input has shape (W,H,3), wrap for PyTorch convolutions
    #     obs_shape = env.observation_space.shape
    #     if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
    #         env = TransposeImage(env)
    #
    #     return env
    #
    # return _thunk


def wrap_initial_policies(envs, device, initial_policies):
    if initial_policies:
        curr_ip, ob_rms, more_ips = initial_policies
        envs = wrap_initial_policies(envs, device, more_ips)
        return ResidualVecEnvWrapper(envs, curr_ip, ob_rms, device)
    return envs


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep, device,
                  allow_early_resets, initial_policies, num_frame_stack=None, show=False,
                  no_norm=False, pose_estimator=None, image_ips=None, e2e=False):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets, show)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if not e2e:
        envs = wrap_initial_policies(envs, device, initial_policies)

    if pose_estimator is not None or e2e:
        # Two separate layers as they may have their uses separately
        envs = ImageObsVecEnvWrapper(envs)

    if len(envs.observation_space.shape) == 1 and not no_norm:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if pose_estimator is not None:
        envs = PoseEstimatorVecEnvWrapper(envs, pose_estimator, device, abs_to_rel=True)
        envs = wrap_initial_policies(envs, device, image_ips)
    if e2e:
        envs = wrap_initial_policies(envs, device, initial_policies)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif not (pose_estimator or e2e) and len(envs.observation_space.shape) == 3:
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
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

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
