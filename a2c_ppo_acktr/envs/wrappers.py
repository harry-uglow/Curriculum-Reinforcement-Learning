from baselines.common.vec_env import VecEnvWrapper
from gym import spaces


class ImageObsVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        venv_obs_space = venv.observation_space
        observation_space = spaces.Tuple((venv_obs_space, spaces.Box(
            0, 255, [venv.res[0], venv.res[1], 3],
            dtype=venv.observation_space.dtype)))
        super().__init__(venv, observation_space)

    def step_async(self, actions):
        super().step_async(actions)

    def reset(self):
        pass

    def step_wait(self):
        pass