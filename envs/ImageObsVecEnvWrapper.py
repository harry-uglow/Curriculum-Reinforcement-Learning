import numpy as np
from baselines.common.vec_env import VecEnvWrapper
from gym import spaces


# Swap out state observation for image
class ImageObsVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, res):
        observation_space = spaces.Box(0, 255, [3, *res], dtype=venv.observation_space.dtype)
        super().__init__(venv, observation_space)
        self.curr_state_obs = None

    def reset(self):
        self.curr_state_obs = self.venv.reset()

    def step_wait(self):
        self.curr_state_obs, rew, done, info = self.venv.step_wait()
        return rew, done, info


class SimImageObsVecEnvWrapper(ImageObsVecEnvWrapper):
    def __init__(self, venv):
        res = venv.get_images(mode='activate')[0]
        super().__init__(venv, res)

    def reset(self):
        super(SimImageObsVecEnvWrapper, self).reset()
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return image_obs

    def step_wait(self):
        rew, done, info = super(SimImageObsVecEnvWrapper, self).step_wait()
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return image_obs, rew, done, info


class RealImageObsVecEnvWrapper(ImageObsVecEnvWrapper):
    def __init__(self, venv, res, camera):
        super().__init__(venv, res)
        self.cam = camera

    def reset(self):
        super(RealImageObsVecEnvWrapper, self).reset()
        image_obs = self.cam.get_image()
        return image_obs

    def step_wait(self):
        rew, done, info = super(RealImageObsVecEnvWrapper, self).step_wait()
        image_obs = self.cam.get_image()
        return image_obs, rew, done, info


def get_image_obs_wrapper(venv):
    if isinstance(venv, ImageObsVecEnvWrapper):
        return venv
    elif hasattr(venv, 'venv'):
        return get_image_obs_wrapper(venv.venv)
    return None
