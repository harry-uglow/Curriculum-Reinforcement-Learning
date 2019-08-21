import numpy as np
import vrep
from envs.DishRackEnv import DishRackEnv
from envs.VrepEnv import catch_errors

rack_lower = [-0.05, (-0.6), -0.25]  # x, y, rotation
rack_upper = [0.15, (-0.45), 0.25]


class DRNoWaypointEnv(DishRackEnv):

    def step(self, a):
        self.target_point = a
        dist = self.get_distance(self.target_handle, self.plate_handle)
        orientation_diff = np.abs(self.get_plate_orientation()).sum()
        rew_collision = - int(catch_errors(vrep.simxReadCollision(
            self.cid, self.collision_handle, vrep.simx_opmode_blocking)))

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_ctrl = - np.square(np.abs(self.target_point).mean())
        rew_orientation = - orientation_diff / max(dist, 0.11)  # Radius = 0.11
        rew = 0.01 * (rew_dist + rew_ctrl + 0.04 * rew_orientation + 0.1 * rew_collision)

        return ob, rew, done, dict(rew_dist=rew_dist, rew_ctrl=rew_ctrl,
                                   rew_orientation=rew_orientation, rew_collision=rew_collision)


class DRNonRespondableEnv(DishRackEnv):

    def step(self, a):
        self.target_point = a
        dist = self.get_distance(self.target_handle, self.plate_handle)
        orientation_diff = np.abs(self.get_plate_orientation()).sum()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - dist
        rew_ctrl = - np.square(np.abs(self.target_point).mean())
        rew_orientation = - orientation_diff / max(dist, 0.11)  # Radius = 0.11
        rew = 0.01 * (rew_dist + rew_ctrl + 0.1 * rew_orientation)

        return ob, rew, done, dict(rew_dist=rew_dist, rew_orientation=rew_orientation)
