import os

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.envs.SawyerEnv import SawyerEnv
from a2c_ppo_acktr.envs.VrepEnv import catch_errors

np.set_printoptions(precision=2, linewidth=200)  # DEBUG

rack_lower = [-0.05, (-0.6), -0.25]  # x, y, rotation
rack_upper = [0.15, (-0.45), 0.25]


class DishRackEnv(SawyerEnv):
    scene_path = 'dish_rack'
    observation_space = spaces.Box(np.array([-3.] * 7 + rack_lower),
                                   np.array([3.] * 7 + rack_upper), dtype=np.float32)
    timestep = 0
    metadata = {'render.modes': ['human', 'rgb_array', 'activate']}
    max_cam_displace = 0.05
    max_cam_rotation = 0.05  # ~2.9 deg
    max_light_displace = 0.5

    # VISION PLACEHOLDERS
    vis_mode = False
    vis_handle = None
    res = None
    plate_obj_handle = None
    cloth_handle = None
    init_cam_pos = None
    init_cam_rot = None
    init_plate_color = None
    init_rack_color = None
    init_cloth_color = None
    light_handles = None
    light_poss = None

    def __init__(self, *args):
        super().__init__(self.scene_path, *args)

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
        self.rack_pos[1] = self.np_random.uniform(rack_lower[1], rack_upper[1])
        self.rack_rot[0] = self.np_random.uniform(rack_lower[2], rack_upper[2])
        vrep.simxSetObjectPosition(self.cid, self.rack_handle, -1, self.rack_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.rack_handle, self.rack_rot_ref, self.rack_rot,
                                      vrep.simx_opmode_blocking)
        self.timestep = 0

        if self.vis_mode:
            # VARY COLORS
            plate_color = self.np_random.normal(loc=self.init_plate_color, scale=0.05)
            rack_color = self.np_random.normal(loc=self.init_rack_color, scale=0.05)
            cloth_color = self.np_random.normal(loc=self.init_cloth_color, scale=0.05)
            self.call_lua_function('set_color', ints=[self.plate_obj_handle], floats=plate_color)
            self.call_lua_function('set_color', ints=[self.rack_handle], floats=rack_color)
            self.call_lua_function('set_color', ints=[self.cloth_handle], floats=cloth_color)
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
            for handle, pos, displace in zip(self.light_handles, self.light_poss,
                                             light_displacement):
                vrep.simxSetObjectPosition(self.cid, handle, -1, pos + displace,
                                           vrep.simx_opmode_blocking)

        return self._get_obs()

    def _get_obs(self):
        joint_obs = super(DishRackEnv, self)._get_obs()

        return np.concatenate((joint_obs, self.get_position(self.target_handle)[:-1],
                               [self.rack_rot[0]]))

    def get_plate_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.plate_handle, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._read_vision_sensor()
        elif mode == 'mask':
            mask = self._read_vision_sensor(grayscale=True)
            mask[mask > 0] = 255
            return mask
        elif mode == 'human':
            return  # Human rendering is automatically handled by headless mode.
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
        self.plate_obj_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Plate",
                                                                      vrep.simx_opmode_blocking))
        self.cloth_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Cloth",
                                                                  vrep.simx_opmode_blocking))
        self.init_cam_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.vis_handle, -1,
                                                                    vrep.simx_opmode_blocking))
        self.init_cam_rot = catch_errors(
            vrep.simxGetObjectOrientation(self.cid, self.vis_handle, -1, vrep.simx_opmode_blocking))
        self.init_plate_color = self.call_lua_function('get_color', ints=[self.plate_obj_handle])[1]
        self.init_rack_color = self.call_lua_function('get_color', ints=[self.rack_handle])[1]
        self.init_cloth_color = self.call_lua_function('get_color', ints=[self.cloth_handle])[1]
        self.light_handles = [catch_errors(vrep.simxGetObjectHandle(self.cid, f'LocalLight{c}',
                                                                    vrep.simx_opmode_blocking))
                              for c in ['A', 'B', 'C', 'D']]
        self.light_poss = [catch_errors(vrep.simxGetObjectPosition(self.cid, handle, -1,
                                                                   vrep.simx_opmode_blocking))
                           for handle in self.light_handles]
