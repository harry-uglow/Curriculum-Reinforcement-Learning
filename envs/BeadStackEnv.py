import glob
import os

import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv
from envs.VrepEnv import catch_errors
import math

np.set_printoptions(precision=2, linewidth=200)  # DEBUG

toy_lower = np.array([-0.05, (-0.6), -0.25])  # x, y, rotation
toy_upper = np.array([0.15, (-0.45), 0.25])

max_displacement = 0.007  # 1.5cm
max_dist = np.linalg.norm([max_displacement]*3)
max_rot = 0.1  # ~5.7 deg


class BeadStackEnv(SawyerEnv):
    observation_space = spaces.Box(np.array([-3.] * 7 + [-math.inf] * 3 + [toy_lower[2]]),
                                   np.array([3.] * 7 + [math.inf] * 3 + [toy_upper[2]]),
                                   dtype=np.float32)
    timestep = 0
    metadata = {'render.modes': ['human', 'rgb_array', 'activate']}

    def __init__(self, *args):
        super().__init__(*args)

        self.ep_len = 32

        self.bead_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Bead_center", vrep.simx_opmode_blocking))
        self.toy_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Toy", vrep.simx_opmode_blocking))
        self.toy_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.toy_handle,
                -1, vrep.simx_opmode_blocking))
        self.toy_rot_ref = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "DefaultOrientation", vrep.simx_opmode_blocking))
        self.toy_rot = catch_errors(vrep.simxGetObjectOrientation(self.cid, self.toy_handle,
                self.toy_rot_ref, vrep.simx_opmode_blocking))
        self.target_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Target", vrep.simx_opmode_blocking))

    def reset(self):
        super(BeadStackEnv, self).reset()
        self.toy_pos[0] = self.np_random.uniform(toy_lower[0], toy_upper[0])
        self.toy_pos[1] = self.np_random.uniform(toy_lower[1], toy_upper[1])
        self.toy_rot[0] = self.np_random.uniform(toy_lower[2], toy_upper[2])
        vrep.simxSetObjectPosition(self.cid, self.toy_handle, -1, self.toy_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.toy_handle, self.toy_rot_ref, self.toy_rot,
                                      vrep.simx_opmode_blocking)
        self.timestep = 0

        return self._get_obs()

    def _get_obs(self):
        joint_obs = super(BeadStackEnv, self)._get_obs()
        pos_vector = self.get_position(self.target_handle) - self.get_position(self.bead_handle)

        return np.concatenate((joint_obs, pos_vector, self.toy_rot[:1]))

    def get_bead_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.bead_handle, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])


class BSSparseEnv(BeadStackEnv):

    def step(self, a):
        self.target_velocities = a
        displacement = np.abs(self.get_vector(self.target_handle, self.bead_handle))
        orientation_diff = np.abs(self.get_bead_orientation())

        rew_success = 0.1 if np.all(orientation_diff <= max_rot) and \
                             np.all(displacement <= max_displacement) else 0
        dist = np.linalg.norm(displacement)
        rew = rew_success * (1 - dist / max_dist)

        self.timestep += 1
        if self.vis_mode:
            self.randomise_domain()
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, rew, done, dict(rew_success=rew_success)


class BSDenseEnv(BeadStackEnv):
    scene_path = 'dish_rack_nr'

    def step(self, a):
        self.target_velocities = a
        dist = self.get_distance(self.target_handle, self.bead_handle)
        orientation_diff = np.abs(self.get_bead_orientation()).sum()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_ctrl = - np.square(np.abs(self.target_velocities).mean())
        rew_orientation = - orientation_diff / max(dist, 0.11)  # Radius = 0.11
        rew = 0.01 * (rew_dist + rew_ctrl + 0.1 * rew_orientation)

        return ob, rew, done, dict(rew_dist=rew_dist, rew_orientation=rew_orientation)
