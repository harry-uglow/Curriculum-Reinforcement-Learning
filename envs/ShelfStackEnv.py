import glob
import os

import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv
from envs.VrepEnv import catch_errors
import math

toy_lower = np.array([-0.1, (-0.75), -0.25])  # x, y, rotation
toy_upper = np.array([0.1, (-0.55), 0.25])

max_displacement = 0.015  # 1.5cm
max_rot = 0.1  # ~5.7 deg


class ShelfStackEnv(SawyerEnv):
    observation_space = spaces.Box(np.array([-3.] * 7 + [-math.inf] * 3),
                                   np.array([3.] * 7 + [math.inf] * 3),
                                   dtype=np.float32)
    timestep = 0

    def __init__(self, *args):
        super().__init__(*args, random_joints=False)

        self.ep_len = 100

        self.mug_h = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Mug_center", vrep.simx_opmode_blocking))
        self.target_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Target", vrep.simx_opmode_blocking))

    def reset(self):
        super(ShelfStackEnv, self).reset()
        self.timestep = 0

        return self._get_obs()

    def _get_obs(self):
        joint_obs = super(ShelfStackEnv, self)._get_obs()
        pos_vector = self.get_position(self.target_handle) - self.get_position(self.mug_h)

        return np.concatenate((joint_obs, pos_vector))

    def get_mug_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.mug_h, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])


class SSSparseEnv(ShelfStackEnv):

    def step(self, a):
        self.target_point = a
        displacement = np.abs(self.get_vector(self.target_handle, self.mug_h))
        orientation_diff = np.abs(self.get_mug_orientation())

        rew_success = 0.1 if np.all(orientation_diff <= max_rot) and \
                             np.all(displacement <= max_displacement) else 0
        rew = rew_success

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, rew, done, dict(rew_success=rew_success)


class SSDenseEnv(ShelfStackEnv):

    def step(self, a):
        self.target_point = a
        dist = self.get_distance(self.target_handle, self.mug_h)
        orientation_diff = np.abs(self.get_mug_orientation()).sum()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_ctrl = - np.square(np.abs(self.target_point).mean())
        rew_orientation = - orientation_diff / max(dist, 0.04)  # Radius = 0.04
        rew = 0.1 * (rew_dist + rew_ctrl + 0.05 * rew_orientation)

        return ob, rew, done, dict(rew_dist=rew_dist, rew_orientation=rew_orientation)
