from baselines.common.vec_env import SubprocVecEnv


class RenderVecEnv(SubprocVecEnv):
    def get_images(self):


    def render(self, mode='rgb_array'):
        return super().render(mode)