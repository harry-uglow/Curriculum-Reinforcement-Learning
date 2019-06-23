import os

import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv

from envs.VrepEnv import check_for_errors, catch_errors

dir_path = os.getcwd()

cube_lower = np.array([0.15, (-0.35), 0.025])
cube_upper = np.array([0.45, (-0.65), 0.5])

max_displacement = 0.03  # 3cm


class ReachOverWallEnv(SawyerEnv):

    scene_path = dir_path + '/reach_over_wall.ttt'
    observation_space = spaces.Box(np.array([0] * 11), np.array([1] * 11), dtype=np.float32)
    timestep = 0

    def __init__(self, *args):
        super().__init__(*args)

        self.ep_len = 100

        return_code, self.end_handle = vrep.simxGetObjectHandle(self.cid,
                "Waypoint_tip", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        self.end_pose = self.get_end_pose()
        return_code, self.target_handle = vrep.simxGetObjectHandle(self.cid,
                "Sphere", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        return_code, self.target_pos = vrep.simxGetObjectPosition(self.cid, self.target_handle,
                -1, vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        return_code, self.wall_handle = vrep.simxGetObjectHandle(self.cid,
                "Wall", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        self.wall_pos = vrep.simxGetObjectPosition(self.cid, self.wall_handle,
                                                   -1, vrep.simx_opmode_blocking)[1]
        self.init_wall_rot = vrep.simxGetObjectOrientation(self.cid,
                self.wall_handle, -1, vrep.simx_opmode_blocking)[1]
        self.wall_orientation = self.init_wall_rot
        self.collision_handle = catch_errors(vrep.simxGetCollisionHandle(self.cid, "Collision",
                                                                         vrep.simx_opmode_blocking))
        self.collided = False

    def reset(self):
        super(ReachOverWallEnv, self).reset()
        self.target_pos[0] = self.np_random.uniform(cube_lower[0], cube_upper[0])
        self.target_pos[1] = self.np_random.uniform(cube_lower[1], cube_upper[1])
        vrep.simxSetObjectPosition(self.cid, self.target_handle, -1, self.target_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.cid, self.wall_handle, -1, self.wall_pos,
                                  vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.wall_handle, -1, self.init_wall_rot,
                                     vrep.simx_opmode_blocking)
        self.timestep = 0
        self.collided = False

        return self._get_obs()

    def step(self, a):
        self.target_velocities = a  # Residual RL
        vec = self.get_end_pose() - self.target_pos

        self.timestep += 1
        self.update_sim()

        self.wall_orientation = vrep.simxGetObjectOrientation(self.cid, self.wall_handle, -1,
                                                              vrep.simx_opmode_blocking)[1]
        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(self.target_velocities).mean()
        reward_obstacle = - np.abs(self.wall_orientation).sum()
        reward = 0.01 * (reward_dist + 0.1 * reward_ctrl + 0.1 * reward_obstacle)

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      reward_obstacle=reward_obstacle)

    def _get_obs(self):
        joint_obs = super(ReachOverWallEnv, self)._get_obs()
        pos_vector = self.get_position(self.target_handle) - self.get_position(self.end_handle)

        return np.concatenate((joint_obs, pos_vector, [self.wall_pos[0]]))

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1,
                                          vrep.simx_opmode_blocking)[1]
        return np.array(pose)


class ROWSparseEnv(ReachOverWallEnv):

    def step(self, a):
        self.target_velocities = a  # Residual RL
        displacement = np.abs(self.get_vector(self.target_handle, self.end_handle))

        rew_success = 0.1 if np.all(displacement <= max_displacement) else 0
        if catch_errors(vrep.simxReadCollision(self.cid, self.collision_handle,
                                               vrep.simx_opmode_blocking)):
            self.collided = True
        reward_obstacle = - 0.05 if self.collided else 0
        rew = rew_success + reward_obstacle

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        return ob, rew, done, dict(rew_success=rew_success, reward_obstacle=reward_obstacle)


class ReachNoWallEnv(ROWSparseEnv):

    def step(self, a):
        self.target_velocities = a
        vec = self.get_end_pose() - self.target_pos

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(self.target_velocities).mean()
        reward = 0.01 * (reward_dist + reward_ctrl)

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

