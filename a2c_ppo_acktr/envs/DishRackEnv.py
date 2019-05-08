import os

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.envs.SawyerEnv import SawyerEnv
from a2c_ppo_acktr.envs.VrepEnv import catch_errors

np.set_printoptions(precision=2, linewidth=200)  # DEBUG
dir_path = os.getcwd()

rack_lower = np.array([-0.05, (-0.45)])
rack_upper = np.array([0.15, (-0.6)])
max_rack_rot = 0.25


class DishRackEnv(SawyerEnv):

    scene_path = dir_path + '/dish_rack.ttt'
    observation_space = spaces.Box(np.array([0] * 10), np.array([1] * 10), dtype=np.float32)
    timestep = 0

    def __init__(self, seed, rank, headless, ep_len=64):
        super().__init__(seed, rank, self.scene_path, headless)

        self.ep_len = ep_len

        self.plate_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Plate_center", vrep.simx_opmode_blocking))
        self.rack_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "DishRack", vrep.simx_opmode_blocking))
        self.rack_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.rack_handle,
                -1, vrep.simx_opmode_blocking))
        self.rack_rot_ref = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "DefaultOrientation", vrep.simx_opmode_blocking))
        self.rack_rot = catch_errors(vrep.simxGetObjectOrientation(self.cid, self.rack_handle,
                self.rack_rot_ref, vrep.simx_opmode_blocking))
        self.collision_handle = catch_errors(vrep.simxGetCollisionHandle(self.cid,
                "Collision", vrep.simx_opmode_blocking))
        self.target_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Target", vrep.simx_opmode_blocking))
        self.target_pos = self.get_target_pos()

    def reset(self):
        super(DishRackEnv, self).reset()
        self.rack_pos[0] = self.np_random.uniform(rack_lower[0], rack_upper[0])
        self.rack_pos[1] = self.np_random.uniform(rack_lower[1], rack_upper[1])
        self.rack_rot[0] = self.np_random.uniform(-max_rack_rot, max_rack_rot)
        vrep.simxSetObjectPosition(self.cid, self.rack_handle, -1, self.rack_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.rack_handle, self.rack_rot_ref, self.rack_rot,
                                      vrep.simx_opmode_blocking)
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        self.target_velocities = a
        dist = np.linalg.norm(self.get_plate_pos() - self.target_pos)
        orientation_diff = np.abs(self.get_plate_orientation()).sum()
        # rew_collision = - int(catch_errors(vrep.simxReadCollision(
        #     self.cid, self.collision_handle, vrep.simx_opmode_blocking)))

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_ctrl = - np.square(np.abs(self.target_velocities).mean())
        rew_orientation = - orientation_diff / max(dist, 0.11)  # Radius = 0.11
        rew = 0.01 * (rew_dist + 2 * rew_ctrl + 0.05 * rew_orientation)  # FIXME: No rack reward

        return ob, rew, done, dict(rew_dist=rew_dist, rew_ctrl=rew_ctrl,
                                   rew_orientation=rew_orientation,)

    def _get_obs(self):
        joint_obs = super(DishRackEnv, self)._get_obs()
        self.target_pos = self.get_target_pos()

        return np.concatenate((joint_obs, self.target_pos[:-1], [self.rack_rot[0]]))

    def get_plate_pos(self):
        pose = catch_errors(vrep.simxGetObjectPosition(
            self.cid, self.plate_handle, -1, vrep.simx_opmode_blocking))
        return np.array(pose)

    def get_target_pos(self):
        pose = catch_errors(vrep.simxGetObjectPosition(
            self.cid, self.target_handle, -1, vrep.simx_opmode_blocking))
        return np.array(pose)

    def get_plate_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.plate_handle, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])
