import os

import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv

from envs.VrepEnv import check_for_errors

dir_path = os.getcwd()

cube_lower = np.array([0.125, -0.125])
cube_upper = np.array([0.7, -0.7])


class SawyerReacherEnv(SawyerEnv):

    observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                   np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                   dtype=np.float32)
    scene_path = dir_path + '/reacher.ttt'
    target_pose = np.array([0.3, -0.3, 0.025])
    timestep = 0

    def __init__(self, seed, rank, headless, ep_len=64):
        super().__init__(seed, rank, self.scene_path, headless)

        self.ep_len = ep_len

        return_code, self.end_handle = vrep.simxGetObjectHandle(self.cid,
                "BaxterGripper_centerJoint", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        _, self.target_handle = vrep.simxGetObjectHandle(self.cid,
                "Cube", vrep.simx_opmode_blocking)

    def reset(self):
        super(SawyerReacherEnv, self).reset()
        self.target_pose[0] = self.np_random.uniform(cube_lower[0], cube_upper[0])
        self.target_pose[1] = self.np_random.uniform(cube_lower[1], cube_upper[1])
        vrep.simxSetObjectPosition(self.cid, self.target_handle, -1, self.target_pose,
                                   vrep.simx_opmode_blocking)

        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        self.target_point = a
        vec = self.get_end_pose() - self.target_pose
        reward_dist = - np.linalg.norm(vec) / 100.

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_ctrl = - np.square(self.target_point).mean() / 100.
        reward = reward_dist + reward_ctrl

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def _get_obs(self):
        joint_obs = super(SawyerReacherEnv, self)._get_obs()
        return np.append(joint_obs, self.target_pose)

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1, vrep.simx_opmode_blocking)[1]
        return np.array(pose)
