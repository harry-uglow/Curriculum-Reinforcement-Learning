import os

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.envs.DishRackEnv import DishRackEnv
from a2c_ppo_acktr.envs.SawyerEnv import SawyerEnv
from a2c_ppo_acktr.envs.VrepEnv import catch_errors

np.set_printoptions(precision=2, linewidth=200)  # DEBUG
dir_path = os.getcwd()

max_dist = 0.015  # 1.5cm
max_rot = 0.1  # ~5.7 deg


class DRSparseEnv(DishRackEnv):
    scene_path = 'dish_rack_pr_14'

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
