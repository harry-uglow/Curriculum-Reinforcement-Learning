import os

import numpy as np
from gym import spaces
import vrep
from envs.GoalDrivenEnv import GoalDrivenEnv

from envs.VrepEnv import check_for_errors

dir_path = os.getcwd()

cube_lower = np.array([0.15, (-0.35), 0.025])
cube_upper = np.array([0.45, (-0.65), 0.5])

max_displacement = 0.03  # 3cm


class ReachOverWallEnv(GoalDrivenEnv):

    scene_path = dir_path + '/reach_over_wall.ttt'
    observation_space = spaces.Box(np.array([0] * 11), np.array([1] * 11), dtype=np.float32)

    def __init__(self, *args):
        super().__init__(*args)

        self.ep_len = 128

        return_code, self.sphere_handle = vrep.simxGetObjectHandle(self.cid,
                "Sphere", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        return_code, self.wall_handle = vrep.simxGetObjectHandle(self.cid,
                "Wall", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        self.wall_pos = vrep.simxGetObjectPosition(self.cid, self.wall_handle,
                                                   -1, vrep.simx_opmode_blocking)[1]
        self.init_wall_rot = vrep.simxGetObjectOrientation(self.cid,
                self.wall_handle, -1, vrep.simx_opmode_blocking)[1]
        self.wall_orientation = self.init_wall_rot

    def reset(self):
        super(ReachOverWallEnv, self).reset()
        self.target_pos[0] = self.np_random.uniform(cube_lower[0], cube_upper[0])
        self.target_pos[1] = self.np_random.uniform(cube_lower[1], cube_upper[1])
        vrep.simxSetObjectPosition(self.cid, self.sphere_handle, -1, self.target_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.cid, self.wall_handle, -1, self.wall_pos,
                                  vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.wall_handle, -1, self.init_wall_rot,
                                     vrep.simx_opmode_blocking)

        return self._get_obs()

    def step(self, a):
        self.curr_action = a
        vec = self.subject_pos - self.target_pos

        self.timestep += 1
        self.update_sim()

        self.wall_orientation = vrep.simxGetObjectOrientation(self.cid, self.wall_handle, -1,
                                                              vrep.simx_opmode_blocking)[1]
        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(self.curr_action).mean()
        reward_obstacle = - np.abs(self.wall_orientation).sum()
        reward = 0.01 * (reward_dist + 0.1 * reward_ctrl + 0.1 * reward_obstacle)

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      reward_obstacle=reward_obstacle)

    def _get_obs(self):
        base_obs = super(ReachOverWallEnv, self)._get_obs()
        return np.append(base_obs, [self.wall_pos[0]])


class ROWSparseEnv(ReachOverWallEnv):

    def step(self, a):
        self.curr_action = a
        displacement = np.abs(self.get_vector(self.target_handle, self.sphere_handle))

        rew_success = 0.1 if np.all(displacement <= max_displacement) else 0
        rew = rew_success

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, rew, done, dict(rew_success=rew_success)


class ReachNoWallEnv(ROWSparseEnv):

    def step(self, a):
        self.curr_action = a
        vec = self.subject_pos - self.target_pos

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(self.curr_action).mean()
        reward = 0.01 * (reward_dist + reward_ctrl)

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

