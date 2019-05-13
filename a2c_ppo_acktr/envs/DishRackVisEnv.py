import os

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.envs.DishRackEnv import DishRackEnv
from a2c_ppo_acktr.envs.SawyerEnv import SawyerEnv
from a2c_ppo_acktr.envs.VrepEnv import catch_errors
from matplotlib import image as im


class DishRackVisEnv(DishRackEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}
    max_cam_displace = 0.1

    def __init__(self, *args):
        super().__init__(*args)

        self.vis_handle = catch_errors(vrep.simxGetObjectHandle(self.cid,
                "Vision_sensor", vrep.simx_opmode_blocking))
        self.plate_obj_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Plate",
                                                                      vrep.simx_opmode_blocking))
        self.stand_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Stand",
                                                                  vrep.simx_opmode_blocking))
        self.init_cam_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.vis_handle, -1,
                                                                    vrep.simx_opmode_blocking))
        self.init_plate_color = self.call_lua_function('get_color', ints=[self.plate_obj_handle])[1]
        self.init_rack_color = self.call_lua_function('get_color', ints=[self.rack_handle])[1]
        self.init_stand_color = self.call_lua_function('get_color', ints=[self.stand_handle])[1]

        vrep.simxSetBooleanParameter(self.cid, vrep.sim_boolparam_vision_sensor_handling_enabled,
                                     True, vrep.simx_opmode_blocking)

    def reset(self):
        super(DishRackEnv, self).reset()
        # VARY COLORS
        plate_color = self.np_random.normal(loc=self.init_plate_color, scale=0.05)
        rack_color = self.np_random.normal(loc=self.init_rack_color, scale=0.05)
        stand_color = self.np_random.normal(loc=self.init_stand_color, scale=0.05)
        self.call_lua_function('set_color', ints=[self.plate_obj_handle], floats=plate_color)
        self.call_lua_function('set_color', ints=[self.rack_handle], floats=rack_color)
        self.call_lua_function('set_color', ints=[self.stand_handle], floats=stand_color)
        # VARY CAMERA POSITION
        cam_displacement = self.np_random.uniform(-self.max_cam_displace, self.max_cam_displace, 3)
        vrep.simxSetObjectPosition(self.cid, self.vis_handle, -1,
                                   self.init_cam_pos + cam_displacement, vrep.simx_opmode_blocking)
        # VARY LIGHTING
        self.update_sim()  # Update not usually needed in reset, but we need the camera to render

        return self._get_obs()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            _, _, _, byte_data = self.call_lua_function('get_image')

            # TODO: Get resolution programmatically
            image = np.frombuffer(byte_data, dtype=np.uint8).reshape((32, 32, 3))
            # DEBUG
            return image
        elif mode == 'human':
            return  # Human rendering is automatically handled by headless mode.
            # TODO: Render footage from vision sensor
        else:
            super(DishRackEnv, self).render(mode=mode)
