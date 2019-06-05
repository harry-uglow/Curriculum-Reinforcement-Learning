import os

import numpy as np
from envs.DishRackEnv import DishRackEnv

np.set_printoptions(precision=2, linewidth=200)  # DEBUG
dir_path = os.getcwd()

max_dist = 0.015  # 1.5cm
max_rot = 0.1  # ~5.7 deg


class DRSparseEnv(DishRackEnv):

    def step(self, a):
        self.target_velocities = a
        dist = np.abs(self.get_vector(self.target_handle, self.plate_handle))
        orientation_diff = np.abs(self.get_plate_orientation())

        rew_success = 0.1 if np.all(orientation_diff <= max_rot) and np.all(dist <= max_dist) else 0

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, rew_success, done, dict(rew_success=rew_success)

    def __init__(self, scene_path, *args):
        self.scene_path = scene_path
        super().__init__(*args)

