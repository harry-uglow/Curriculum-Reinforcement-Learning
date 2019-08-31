import os

import numpy as np
from envs.DishRackEnv import DishRackEnv

dir_path = os.getcwd()

max_displacement = 0.015  # 1.5cm
max_dist = np.linalg.norm([max_displacement]*3)
max_rot = 0.1  # ~5.7 deg


class DRSparseEnv(DishRackEnv):

    def step(self, a):
        self.curr_action = a
        displacement = np.abs(self.get_vector(self.target_handle, self.subject_handle))
        orientation_diff = np.abs(self.get_plate_orientation())

        rew = 0.1 if np.all(orientation_diff <= max_rot) and \
                             np.all(displacement <= max_displacement) else 0

        self.timestep += 1
        if self.vis_mode:
            self.randomise_domain()
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, rew, done, dict(rew_success=rew)


class DRDenseEnv(DishRackEnv):

    def step(self, a):
        self.curr_action = a
        displacement = np.abs(self.get_vector(self.target_handle, self.subject_handle))
        dist = np.linalg.norm(displacement)
        orientation_diff = np.abs(self.get_plate_orientation()).sum()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_orientation = - orientation_diff / max(dist, 0.11)  # Radius = 0.11
        rew = 0.01 * (rew_dist + 0.1 * rew_orientation)

        rew_success = 1 if np.all(orientation_diff <= max_rot) and \
                           np.all(displacement <= max_displacement) else 0

        return ob, rew, done, dict(rew_dist=rew_dist, rew_orientation=rew_orientation,
                                   rew_success=rew_success)
