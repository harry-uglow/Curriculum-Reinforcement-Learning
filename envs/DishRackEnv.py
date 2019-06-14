import glob
import os

import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv
from envs.VrepEnv import catch_errors
import math

np.set_printoptions(precision=2, linewidth=200)  # DEBUG

rack_lower = np.array([-0.05, (-0.6), -0.25])  # x, y, rotation
rack_upper = np.array([0.15, (-0.45), 0.25])


class DishRackEnv(SawyerEnv):
    observation_space = spaces.Box(np.array([-3.]*7 + [-math.inf]*3 + [rack_lower[2]]),
                                   np.array([3.]*7 + [math.inf]*3 + [rack_upper[2]]),
                                   dtype=np.float32)
    timestep = 0
    metadata = {'render.modes': ['human', 'rgb_array', 'activate']}
    max_cam_displace = 0.05
    max_cam_rotation = 0.05  # ~2.9 deg
    max_light_displace = 0.5
    max_cloth_rotation = 0.05

    # VISION PLACEHOLDERS
    vis_mode = False
    vis_handle = None
    res = None
    plate_obj_handle = None
    cloth_handle = None
    back_wall_h = None
    init_cloth_pos = None
    init_cloth_rot = None
    init_cam_pos = None
    init_cam_rot = None
    init_colors = []
    init_sawyer_colors = None
    init_plate_color = None
    init_rack_color = None
    init_cloth_color = None
    init_wall_color = None
    light_handles = None
    light_poss = None
    ep_num = 0
    num_randomisation_eps = 8
    left_wall_h = None
    button_base_b_h = None
    button_base_y_h = None
    button_h = None
    stand_h = None
    stand_height = None
    button_pos = None
    block_pos = None
    block_h = None
    sawyer_links = None
    max_button_displacement = 0.03
    init_button_pos = None
    max_height_displacement = 0.01
    init_stand_pos = None
    init_block_pos = None
    max_block_displacement = np.array([0.075, 0.05, 0.025])

    def __init__(self, *args):
        super().__init__(*args)

        self.ep_len = 48

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
            self.randomise_domain()
        return self._get_obs()

    def _get_obs(self):
        joint_obs = super(DishRackEnv, self)._get_obs()
        pos_vector = self.get_position(self.target_handle) - self.get_position(self.plate_handle)
        # pos_vector[2] += 0.05

        return np.concatenate((joint_obs, pos_vector, self.rack_rot[:1]))

    def get_plate_orientation(self):
        orientation = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.plate_handle, self.target_handle, vrep.simx_opmode_blocking))
        return np.array(orientation[:-1])

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
        self.init_cloth_pos = catch_errors(vrep.simxGetObjectPosition(
            self.cid, self.cloth_handle, -1, vrep.simx_opmode_blocking))
        self.init_cloth_rot = catch_errors(vrep.simxGetObjectOrientation(
            self.cid, self.cloth_handle, -1, vrep.simx_opmode_blocking))
        self.back_wall_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "WallB",
                                                                 vrep.simx_opmode_blocking))
        self.left_wall_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "WallL",
                                                                 vrep.simx_opmode_blocking))
        self.button_base_b_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "ButtonBaseB",
                                                                     vrep.simx_opmode_blocking))
        self.button_base_y_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "ButtonBaseY",
                                                                     vrep.simx_opmode_blocking))
        self.button_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "Button",
                                                              vrep.simx_opmode_blocking))
        self.stand_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "Stand",
                                                             vrep.simx_opmode_blocking))
        self.block_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "Cuboid",
                                                             vrep.simx_opmode_blocking))
        self.sawyer_links = [
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link0_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link1_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link2_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link3_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link4_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link5_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_link6_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "Sawyer_head_visible",
                                                  vrep.simx_opmode_blocking))
        ]
        self.gripper_h = catch_errors(vrep.simxGetObjectHandle(self.cid, "BaxterGripper_visible",
                                                               vrep.simx_opmode_blocking))
        self.gripper_links = [

            catch_errors(vrep.simxGetObjectHandle(self.cid, "BaxterGripper_rightFinger_visible",
                                                  vrep.simx_opmode_blocking)),
            catch_errors(vrep.simxGetObjectHandle(self.cid, "BaxterGripper_leftFinger_visible",
                                                  vrep.simx_opmode_blocking)),
        ]
        self.init_cam_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.vis_handle, -1,
                                                                    vrep.simx_opmode_blocking))
        self.init_cam_rot = catch_errors(
            vrep.simxGetObjectOrientation(self.cid, self.vis_handle, -1, vrep.simx_opmode_blocking))
        self.init_button_pos = catch_errors(
            vrep.simxGetObjectPosition(self.cid, self.button_base_b_h, self.block_h,
                                       vrep.simx_opmode_blocking))
        self.init_stand_pos = catch_errors(
            vrep.simxGetObjectPosition(self.cid, self.stand_h, -1, vrep.simx_opmode_blocking))
        self.init_block_pos = catch_errors(
            vrep.simxGetObjectPosition(self.cid, self.block_h, -1, vrep.simx_opmode_blocking))

        def init_color(handle, scale):
            return [(handle, self.call_lua_function('get_color', ints=[handle])[1], scale)]

        def init_named_color(handle, scale, name='SAWYER_RED'):
            return [(handle, self.call_lua_function('get_sawyer_color', ints=[handle],
                                                    strings=[name])[1], scale)]

        self.init_colors += init_color(self.plate_obj_handle, 0.05)
        self.init_colors += init_color(self.rack_handle, 0.05)
        self.init_colors += init_color(self.cloth_handle, 0.05)
        self.init_colors += init_color(self.back_wall_h, 0.02)
        self.init_colors += init_color(self.left_wall_h, 0.02)
        self.init_colors += init_color(self.button_base_b_h, 0.02)
        self.init_colors += init_color(self.button_base_y_h, 0.02)
        self.init_colors += init_color(self.button_h, 0.02)
        self.init_colors += init_color(self.stand_h, 0.02)
        self.init_colors += init_color(self.block_h, 0.02)
        self.init_colors += [init_color(handle, 0.02)[0] for handle in self.gripper_links]
        self.init_sawyer_colors = [init_named_color(handle, 0.02)[0] for handle in
                                   self.sawyer_links]
        self.init_baxter_color = init_named_color(self.gripper_h, 0.02, name='BAXTER_RED')[0][1:]

        self.light_handles = [catch_errors(vrep.simxGetObjectHandle(self.cid, f'LocalLight{c}',
                                                                    vrep.simx_opmode_blocking))
                              for c in ['A', 'B', 'C', 'D']]
        self.light_poss = [catch_errors(vrep.simxGetObjectPosition(self.cid, handle, -1,
                                                                   vrep.simx_opmode_blocking))
                           for handle in self.light_handles]

    def randomise_domain(self):
        # VARY COLORS
        colors = [(handle, self.np_random.normal(loc=color, scale=scale))
                  for handle, color, scale in self.init_colors]
        for handle, color in colors:
            self.call_lua_function('set_color', ints=[handle], floats=color)
        for handle, color in [(handle, self.np_random.normal(loc=color, scale=scale))
                              for handle, color, scale in self.init_sawyer_colors]:
            self.call_lua_function('set_color', ints=[handle], floats=color, strings=['SAWYER_RED'])

        self.call_lua_function('set_color', ints=[self.gripper_h],
                               floats=self.np_random.normal(loc=self.init_baxter_color[0],
                                                            scale=self.init_baxter_color[1]),
                               strings=['BAXTER_RED'])

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
        # VARY CLOTH POSE
        cloth_pos = self.init_cloth_pos.copy()
        cloth_pos[1] += - np.abs(self.np_random.normal(0, 0.01))
        vrep.simxSetObjectPosition(self.cid, self.cloth_handle, -1, cloth_pos,
                                   vrep.simx_opmode_blocking)
        cloth_rot = self.init_cloth_rot.copy()
        cloth_rot[2] = self.np_random.uniform(-self.max_cloth_rotation, self.max_cloth_rotation)
        vrep.simxSetObjectOrientation(self.cid, self.cloth_handle, -1, cloth_rot,
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
        # OTHER POSES
        block_displacement = self.np_random.uniform(-self.max_block_displacement,
                                                    self.max_block_displacement)
        vrep.simxSetObjectPosition(self.cid, self.block_h, -1,
                                   self.init_block_pos + block_displacement,
                                   vrep.simx_opmode_blocking)
        button_displacement = np.append(self.np_random.uniform(-self.max_button_displacement,
                                                               self.max_button_displacement, 2),
                                        [0])
        vrep.simxSetObjectPosition(self.cid, self.button_base_b_h, self.block_h,
                                   self.init_button_pos + button_displacement,
                                   vrep.simx_opmode_blocking)
        stand_height_diff = np.append([0, 0], self.np_random.uniform(-self.max_height_displacement,
                                                                     self.max_height_displacement,
                                                                     1))
        vrep.simxSetObjectPosition(self.cid, self.stand_h, -1,
                                   self.init_stand_pos + stand_height_diff,
                                   vrep.simx_opmode_blocking)
