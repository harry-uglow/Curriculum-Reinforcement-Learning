from __future__ import division
from __future__ import absolute_import
import numpy as np

import vrep
from envs.DishRackEnv import DishRackEnv
from envs.VrepEnv import catch_errors

max_rot = 0.1  # ~5.7 deg


# Static target with waypoints
class DRWaypointEnv(DishRackEnv):
    reached_waypoint = False

    def __init__(self, scene_path, *args):
        super(DRWaypointEnv, self).__init__(u'dish_rack', *args)
        self.ep_len = 64
        self.waypoint_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, u"Waypoint",
                                                                     vrep.simx_opmode_blocking))

    def reset(self):
        self.reached_waypoint = False
        self.call_lua_function(u'set_joint_angles', ints=self.init_config_tree,
                               floats=self.init_joint_angles)
        self.target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        self.target_velocities = a
        plate_trg = self.get_distance(self.target_handle, self.plate_handle)
        plate_way = self.get_distance(self.waypoint_handle, self.plate_handle)
        way_trg = self.get_distance(self.target_handle, self.waypoint_handle)

        if not self.reached_waypoint and plate_way < 0.05:
            self.reached_waypoint = True

        orientation_diff = np.abs(self.get_plate_orientation()).sum()

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        rew_dist = - (plate_trg if self.reached_waypoint else plate_way + way_trg)
        rew_ctrl = - np.square(np.abs(self.target_velocities).mean())
        rew_orientation = - orientation_diff / max(plate_trg, 0.11)  # Radius = 0.11
        rew = 0.01 * (rew_dist + rew_ctrl + 0.04 * rew_orientation)

        return ob, rew, done, dict(rew_dist=rew_dist, rew_ctrl=rew_ctrl,
                                   rew_orientation=rew_orientation)
