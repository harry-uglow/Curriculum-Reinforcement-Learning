import numpy as np
from baselines.common.vec_env import SubprocVecEnv


class RenderVecEnv(SubprocVecEnv):

    def render(self, mode='rgb_array'):
        imgs = self.get_images()
        bigimg = np.asarray(imgs)
        if mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
