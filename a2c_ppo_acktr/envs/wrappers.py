import numpy as np
import torch
from baselines.common.vec_env import VecEnvWrapper
from gym import spaces, ActionWrapper

from a2c_ppo_acktr.envs.ResidualVecEnvWrapper import get_residual_layers
from im2state.utils import unnormalise_y


class ImageObsVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        res = np.array(venv.get_images()[0].shape)
        observation_space = spaces.Box(0, 255, res[[2, 0, 1]], dtype=venv.observation_space.dtype)
        super().__init__(venv, observation_space)
        self.curr_state_obs = None

    def reset(self):
        self.curr_state_obs = self.venv.reset()
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return image_obs

    # Swap out state for image
    def step_wait(self):
        self.curr_state_obs, rew, done, info = self.venv.step_wait()
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return image_obs, rew, done, info


class PoseEstimatorVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, pose_estimator, device):
        super().__init__(venv)
        self.image_obs_wrapper = get_image_obs_wrapper(venv)
        assert self.image_obs_wrapper is not None
        self.estimator = pose_estimator
        self.estimator.eval()
        self.estimator.to(device)
        self.policy_layers = get_residual_layers(venv)
        self.state_obs_space = self.policy_layers[0].observation_space
        self.low = self.state_obs_space.low[pose_estimator.state_to_estimate]
        self.high = self.state_obs_space.high[pose_estimator.state_to_estimate]
        self.curr_image = None

    def step_async(self, actions):
        with torch.no_grad():
            estimation = unnormalise_y(self.estimator(self.curr_image).cpu().numpy(),
                                       self.low,
                                       self.high)
            obs = self.image_obs_wrapper.curr_state_obs
            obs[:, self.estimator.state_to_estimate] = estimation
            for policy in self.policy_layers:
                policy.curr_obs = policy.normalize_obs(obs)
        self.venv.step_async(actions)

    def step_wait(self):
        self.curr_image, rew, done, info = self.venv.step_wait()
        return self.curr_image, rew, done, info

    def reset(self):
        self.curr_image = self.venv.reset()
        return self.curr_image


def get_image_obs_wrapper(venv):
    if isinstance(venv, ImageObsVecEnvWrapper):
        return venv
    elif hasattr(venv, 'venv'):
        return get_image_obs_wrapper(venv.venv)
    return None


class ScaleActions(ActionWrapper):
    def __init__(self, env):
        super(ScaleActions, self).__init__(env)

    def action(self, action):
        action_range = self.action_space.high - self.action_space.low
        return (np.tanh(action) + 1) / 2 * action_range + self.action_space.low
