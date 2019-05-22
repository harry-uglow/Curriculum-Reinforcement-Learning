import numpy as np

import vrep
from a2c_ppo_acktr.envs.DishRackEnv import DishRackEnv
from a2c_ppo_acktr.envs.VrepEnv import catch_errors

max_rot = 0.1  # ~5.7 deg

class DRWaypointEnv(DishRackEnv):
    reached_waypoint = False

    def __init__(self, *args):
        super().__init__(*args)
        self.waypoint_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Waypoint",
                                                                     vrep.simx_opmode_blocking))

    def reset(self):
        self.reached_waypoint = False
        return super(DRWaypointEnv, self).reset()

    def step(self, a):
        self.target_velocities = a
        plate_trg = self.get_distance(self.target_handle, self.plate_handle)
        plate_way = self.get_distance(self.waypoint_handle, self.plate_handle)
        way_trg = self.get_distance(self.target_handle, self.waypoint_handle)

        if not self.reached_waypoint and plate_way < 0.05:
            self.reached_waypoint = True

        orientation_diff = np.abs(self.get_plate_orientation()).sum()
        rew_collision = - int(catch_errors(vrep.simxReadCollision(
            self.cid, self.collision_handle, vrep.simx_opmode_blocking)))

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - (plate_trg if self.reached_waypoint else plate_way + way_trg)
        rew_ctrl = - np.square(np.abs(self.target_velocities).mean())
        rew_orientation = 0.05 if np.all(orientation_diff <= max_rot) else 0
        rew = 0.01 * rew_dist

        return ob, rew, done, dict(rew_dist=rew_dist)
