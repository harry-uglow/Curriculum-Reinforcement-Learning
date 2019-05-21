import numpy as np
import torch
from baselines.common.vec_env import VecEnvWrapper
from gym import spaces

from a2c_ppo_acktr.envs.ResidualVecEnvWrapper import get_residual_layers
from im2state.utils import unnormalise_y


class ImageObsVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        observation_space = spaces.Box(0, 255, [venv.res[0], venv.res[1], 3],
                                       dtype=venv.observation_space.dtype)
        super().__init__(venv, observation_space)
        self.curr_state_obs = None

    # TODO: If this is all that's needed, remove.
    def step_async(self, actions):
        self.venv.step_async(actions)

    # Swap out state for image
    def step_wait(self):
        self.curr_state_obs, rew, done, info = self.venv.step_wait()
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return image_obs, rew, done, info


class PoseEstimatorVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, pose_estimator, state_indices):
        assert isinstance(venv, ImageObsVecEnvWrapper)
        super().__init__(venv)
        self.estimator = pose_estimator
        self.estimator.eval()
        self.policy_layers = get_residual_layers(venv)
        self.state_obs_space = self.policy_layers[0].observation_space
        self.state_to_estimate = state_indices
        self.low = self.state_obs_space.low[self.state_to_estimate]
        self.high = self.state_obs_space.high[self.state_to_estimate]
        self.curr_image = None

    def step_async(self, actions):
        with torch.no_grad():
            estimation = unnormalise_y(self.estimator(self.curr_image).numpy(), self.low, self.high)
            obs = self.venv.curr_state_obs
            obs[:, self.state_to_estimate] = estimation
            for policy in self.policy_layers:
                policy.curr_obs = policy.normalize_obs(obs)
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()
