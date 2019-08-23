import numpy as np
from gym import spaces
import vrep
from envs.GoalDrivenEnv import GoalDrivenEnv
from envs.VrepEnv import catch_errors
import math

trg_lower = np.array([0.85, 0.])  # x, y
trg_upper = np.array([0.95, 0.3])

max_displacement = 0.015  # 1.5cm
max_rot = 0.1  # ~5.7 deg


class ShelfStackEnv(GoalDrivenEnv):
    observation_space = spaces.Box(np.array([-3.] * 7 + [-math.inf] * 3),
                                   np.array([3.] * 7 + [math.inf] * 3),
                                   dtype=np.float32)

    def __init__(self, *args):
        super().__init__(*args, random_joints=False)

        self.ep_len = 100

    def reset(self):
        super(ShelfStackEnv, self).reset()
        self.target_pos[0] = self.np_random.uniform(trg_lower[0], trg_upper[0])
        self.target_pos[1] = self.np_random.uniform(trg_lower[1], trg_upper[1])
        vrep.simxSetObjectPosition(self.cid, self.target_handle, -1, self.target_pos,
                                   vrep.simx_opmode_blocking)

        return self._get_obs()

    def get_mug_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.subject_handle, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])


class SSSparseEnv(ShelfStackEnv):

    def step(self, a):
        self.curr_action = a
        displacement = np.abs(self.get_vector(self.target_handle, self.subject_handle))
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
        self.curr_action = a
        dist = self.get_distance(self.target_handle, self.subject_handle)
        orientation_diff = np.abs(self.get_mug_orientation()).sum()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_ctrl = - np.square(np.abs(self.curr_action).mean())
        rew_orientation = - orientation_diff / max(dist, 0.04)  # Radius = 0.04
        rew = 0.1 * (rew_dist + rew_ctrl + 0.05 * rew_orientation)

        return ob, rew, done, dict(rew_dist=rew_dist, rew_orientation=rew_orientation)
