import glob
import os

import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv
from envs.VrepEnv import catch_errors
import math

rack_lower = np.array([-0.05, (-0.6), -0.25])  # x, y, rotation
rack_upper = np.array([0.15, (-0.45), 0.25])


class DishRackEnv(SawyerEnv):
    observation_space = spaces.Box(np.array([-3.]*7 + [-math.inf]*3 + [rack_lower[2]]),
                                   np.array([3.]*7 + [math.inf]*3 + [rack_upper[2]]),
                                   dtype=np.float32)
    timestep = 0
    metadata = {'render.modes': ['human', 'rgb_array', 'activate']}
    max_cam_displace = 0.05
    max_cam_rotation = 0.2
    max_light_displace = 1.
    max_cloth_rotation = 0.05
    max_height_displacement = 0.02

    # VISION PLACEHOLDERS
    vis_mode = False
    vis_handle = None
    res = None
    plate_handle = None
    cloth_handle = None
    init_cam_pos = None
    init_cam_rot = None
    init_colors = []
    init_plate_color = None
    init_rack_color = None
    init_cloth_color = None
    light_handles = None
    light_poss = None
    light_rots = None
    stand_h = None
    stand_height = None
    init_stand_pos = None

    def __init__(self, *args):
        super().__init__(*args)

        self.ep_len = 32

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

    def reset(self):
        super(DishRackEnv, self).reset()
        self.rack_pos[0] = self.np_random.uniform(rack_lower[0], rack_upper[0])
        self.rack_pos[0] = 0.15
        self.rack_pos[1] = self.np_random.uniform(rack_lower[1], rack_upper[1])
        self.rack_rot[0] = self.np_random.uniform(rack_lower[2], rack_upper[2])
        vrep.simxSetObjectPosition(self.cid, self.rack_handle, -1, self.rack_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.rack_handle, self.rack_rot_ref, self.rack_rot,
                                      vrep.simx_opmode_blocking)
        self.timestep = 0

        if self.vis_mode:
            # OTHER POSES
            stand_height_diff = np.append([0, 0],
                                          self.np_random.uniform(-self.max_height_displacement,
                                                                 self.max_height_displacement,
                                                                 1))
            vrep.simxSetObjectPosition(self.cid, self.stand_h, -1,
                                       self.init_stand_pos + stand_height_diff,
                                       vrep.simx_opmode_blocking)

            self.randomise_domain()
        return self._get_obs()

    def _get_obs(self):
        joint_obs = super(DishRackEnv, self)._get_obs()
        pos_vector = self.get_position(self.target_handle) - self.get_position(self.plate_handle)

        return np.concatenate((joint_obs, pos_vector, self.rack_rot[:1]))

    def get_plate_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.plate_handle, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])

    # Typical render modes are rgb_array and human. Others are abuse of the get_images/render
    # functions for gathering training data from base environments.
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._read_vision_sensor()
        elif mode == 'target':
            pos = self.get_position(self.target_handle)
            return np.append(pos[:-1], self.rack_rot[:1])
        elif mode == 'plate':
            return self.get_position(self.plate_handle)
        elif mode == 'target_height':
            return self.get_position(self.target_handle)[-1:]
        elif mode == 'action':
            return self.target_velocity
        elif mode == 'mask':
            mask = self._read_vision_sensor(grayscale=True)
            mask[mask > 0] = 255
            return mask
        elif mode == 'human':
            return
            # TODO: Render footage from vision sensor
        elif mode == 'activate':
            assert not self.vis_mode
            self.vis_mode = True
            self.setup_vision()
            return self.res
        else:
            super(DishRackEnv, self).render(mode=mode)

    def _read_vision_sensor(self, grayscale=False):
        vrep.simxSetObjectIntParameter(self.cid, self.vis_handle,
                                       vrep.sim_visionintparam_entity_to_render, self.rack_handle
                                       if grayscale else -1, vrep.simx_opmode_blocking)
        _, _, _, byte_data = self.call_lua_function('get_image', ints=[int(grayscale)])
        num_channels = len(byte_data) // (self.res[0] * self.res[1])
        return np.frombuffer(byte_data,
                             dtype=np.uint8).reshape((self.res[0], self.res[1], num_channels))

    def setup_vision(self):
        self.vis_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                                                                "Vision_sensor",
                                                                vrep.simx_opmode_blocking))
        self.res = self.call_lua_function('get_resolution')[0]
        self.plate_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Plate",
                                                                  vrep.simx_opmode_blocking))
        self.cloth_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Cloth",
                                                                  vrep.simx_opmode_blocking))
        self.stand_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "Stand",
                                                             vrep.simx_opmode_blocking))
        self.init_cam_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.vis_handle, -1,
                                                                    vrep.simx_opmode_blocking))
        self.init_cam_rot = catch_errors(
            vrep.simxGetObjectOrientation(self.cid, self.vis_handle, -1, vrep.simx_opmode_blocking))
        self.init_stand_pos = catch_errors(
            vrep.simxGetObjectPosition(self.cid, self.stand_h, -1, vrep.simx_opmode_blocking))

        def init_color(handle, scale):
            return [(handle, self.call_lua_function('get_color', ints=[handle])[1], scale)]

        self.init_colors += init_color(self.plate_handle, 0.025)
        self.init_colors += init_color(self.rack_handle, 0.05)
        self.init_colors += init_color(self.cloth_handle, 0.05)

        self.light_handles = [catch_errors(vrep.simxGetObjectHandle(self.cid, f'LocalLight{c}',
                                                                    vrep.simx_opmode_blocking))
                              for c in ['A', 'B', 'C', 'D']]
        self.light_poss = [catch_errors(vrep.simxGetObjectPosition(self.cid, handle, -1,
                                                                   vrep.simx_opmode_blocking))
                           for handle in self.light_handles]
        self.light_rots = [catch_errors(vrep.simxGetObjectOrientation(self.cid, handle, -1,
                                                                   vrep.simx_opmode_blocking))
                           for handle in self.light_handles]

    def randomise_domain(self):
        # VARY COLORS
        colors = [(handle, self.np_random.normal(loc=color, scale=scale))
                  for handle, color, scale in self.init_colors]
        for handle, color in colors:
            self.call_lua_function('set_color', ints=[handle], floats=color)

        # VARY CAMERA POSE
        cam_displacement = self.np_random.uniform(-self.max_cam_displace,
                                                  self.max_cam_displace, 3)
        vrep.simxSetObjectPosition(self.cid, self.vis_handle, -1,
                                   self.init_cam_pos + cam_displacement,
                                   vrep.simx_opmode_blocking)
        orientation_displacement = ((self.np_random.beta(2, 2, 3) - 0.5) * 2
                                    * self.max_cam_rotation)
        vrep.simxSetObjectOrientation(self.cid, self.vis_handle, -1,
                                      self.init_cam_rot + orientation_displacement,
                                      vrep.simx_opmode_blocking)

        # VARY LIGHTING
        # B and C are support lights and can be disabled.
        enabled = np.random.choice(a=[False, True], size=2)
        for i in range(2):
            f_name = 'enable_light' if enabled[i] else 'disable_light'
            self.call_lua_function(f_name, ints=[self.light_handles[i + 1]])
        light_displacement = self.np_random.uniform(-self.max_light_displace,
                                                    self.max_light_displace, (4, 3))
        orientation_displacement = self.np_random.uniform(-self.max_cam_rotation,
                                                          self.max_cam_rotation, 3)
        for handle, pos, pdisplace, rot, rdisplace in zip(self.light_handles, self.light_poss,
                                                          light_displacement, self.light_rots,
                                                          orientation_displacement):
            vrep.simxSetObjectPosition(self.cid, handle, -1, pos + pdisplace,
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.cid, handle, -1, rot + rdisplace,
                                       vrep.simx_opmode_blocking)
        orientation_displacement = self.np_random.uniform(-self.max_cam_rotation,
                                                            self.max_cam_rotation, 3)
        vrep.simxSetObjectOrientation(self.cid, self.vis_handle, -1,
                                      self.init_cam_rot + orientation_displacement,
                                      vrep.simx_opmode_blocking)
