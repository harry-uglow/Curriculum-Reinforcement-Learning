import os

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.envs.DishRackEnv import DishRackEnv
from a2c_ppo_acktr.envs.SawyerEnv import SawyerEnv
from a2c_ppo_acktr.envs.VrepEnv import catch_errors

np.set_printoptions(precision=2, linewidth=200)  # DEBUG
dir_path = os.getcwd()

max_dist = 0.01  # 1cm
max_rot = 0.1  # ~5.7 deg


class DishRackSparseEnv(DishRackEnv):

    def step(self, a):
        self.target_velocities = a
        dist = np.linalg.norm(self.get_plate_pos() - self.target_pos)
        orientation_diff = self.get_plate_orientation()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_success = 1 if np.all(orientation_diff <= max_rot) or dist <= max_dist else 0

        rew = rew_success

        return ob, rew, done, dict(rew_success=rew_success)
