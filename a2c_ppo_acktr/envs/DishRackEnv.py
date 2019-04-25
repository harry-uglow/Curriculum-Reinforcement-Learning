import os
import torch

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.envs.SawyerEnv import SawyerEnv, normalise_coords, normalise_angles
from a2c_ppo_acktr.residual.ROW_utils import train_initial_policy
from a2c_ppo_acktr.residual.initial_policy_model import InitialPolicy

from a2c_ppo_acktr.envs.VrepEnv import check_for_errors

np.set_printoptions(precision=2, linewidth=200)  # DEBUG
dir_path = os.getcwd()

cube_lower = np.array([0.15, (-0.35), 0.025])
cube_upper = np.array([0.45, (-0.65), 1])


class DishRackEnv(SawyerEnv):

    scene_path = dir_path + '/dish_rack.ttt'
    observation_space = spaces.Box(np.array([0] * 11), np.array([1] * 11), dtype=np.float32)
    timestep = 0

    def __init__(self, seed, rank, initial_policy, headless, ep_len=64):
        super().__init__(seed, rank, self.scene_path, headless)

        self.target_pos = np.array([0.3, -0.5, 0.1])  # TODO: Obtain
        self.waypoint_pos = np.array([0, -0.5, 0.45])  # TODO: Obtain
        self.target_norm = normalise_coords(self.target_pos, cube_lower, cube_upper)
        self.ep_len = ep_len
        self.initial_policy = initial_policy

        return_code, self.end_handle = vrep.simxGetObjectHandle(self.cid,
                "Waypoint_tip", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        self.end_pose = self.get_end_pose()
        return_code, self.target_handle = vrep.simxGetObjectHandle(self.cid,
                "Cube", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        return_code, self.wall_handle = vrep.simxGetObjectHandle(self.cid,
                "Wall", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        self.init_wall_pos = vrep.simxGetObjectPosition(self.cid, self.wall_handle,
                -1, vrep.simx_opmode_blocking)[1]
        self.wall_distance = normalise_coords(self.init_wall_pos[0], lower=0, upper=1)
        self.init_wall_rot = vrep.simxGetObjectOrientation(self.cid,
                self.wall_handle, -1, vrep.simx_opmode_blocking)[1]
        self.wall_orientation = self.init_wall_rot

    def reset(self):
        super(DishRackEnv, self).reset()
        vrep.simxSetObjectPosition(self.cid, self.wall_handle, -1, self.init_wall_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.wall_handle, -1, self.init_wall_rot,
                                      vrep.simx_opmode_blocking)
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        ip_input = torch.Tensor(normalise_angles(self.joint_angles)).reshape((1, self.num_joints))
        with torch.no_grad():
            ip_action = self.initial_policy(ip_input).detach().numpy().flatten()
        self.target_velocities = ip_action + a  # Residual RL
        vec = self.end_pose - self.target_pos
        reward_dist = - np.linalg.norm(vec)

        self.timestep += 1
        self.update_sim()

        self.wall_orientation = vrep.simxGetObjectOrientation(self.cid, self.wall_handle, -1,
                                                              vrep.simx_opmode_blocking)[1]
        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_ctrl = - np.square(self.target_velocities).mean()
        reward_obstacle = - np.abs(self.wall_orientation).sum()
        reward = 0.01 * (reward_dist + reward_ctrl + 0.5 * reward_obstacle)

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      reward_obstacle=reward_obstacle)

    def _get_obs(self):
        joint_obs = super(DishRackEnv, self)._get_obs()
        self.end_pose = self.get_end_pose()

        return np.concatenate((joint_obs, self.target_norm, [self.wall_distance]))

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1,
                                          vrep.simx_opmode_blocking)[1]
        return np.array(pose)


class ROWRandomTargetEnv(DishRackEnv):

    def reset(self):
        self.target_pos[0] = self.np_random.uniform(cube_lower[0], cube_upper[0])
        self.target_pos[1] = self.np_random.uniform(cube_lower[1], cube_upper[1])
        self.target_norm = normalise_coords(self.target_pos, cube_lower, cube_upper)
        vrep.simxSetObjectPosition(self.cid, self.target_handle, -1, self.target_pos,
                                   vrep.simx_opmode_blocking)
        return super(ROWRandomTargetEnv, self).reset()


class ROWEnvInitialiser(DishRackEnv):

    def __init__(self, seed, rank):
        ip = InitialPolicy(self.num_joints, self.num_joints)
        super().__init__(seed, rank, ip, True, ep_len=128)

    def solve_ik(self):
        end = self.get_end_pose()
        end_x = end[0]
        end_z = end[2]

        strings = ['IK_GroupW', 'tip_waypoint'] \
            if end_x < -0.005 and end_z < 0.445 - end_x \
            else ['IK_GroupT', 'tip_target']

        _, path, _, _ = self.call_lua_function('solve_ik', strings=strings)
        num_path_points = len(path) // self.num_joints
        path = np.reshape(path, (num_path_points, self.num_joints))
        distances = np.array([path[i + 1] - path[i]
                              for i in range(0, len(path) - 1)])
        velocities = distances * 20  # Distances should be covered in 0.05s
        return path, velocities

    def get_initial_data(self, num_samples=10, scale=0.01):
        demo_path, demo_velocities = self.get_demo_path()
        id = scale * np.identity(self.num_joints)
        poses = np.array([self.np_random.multivariate_normal(pose, id, num_samples)
                          for pose in demo_path])
        poses = np.reshape(poses, (len(demo_path) * num_samples, self.num_joints))
        pose_list = [demo_path[:-1]]
        velocity_list = [demo_velocities]

        for pose in poses:
            self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=pose)
            path, velocities = self.solve_ik()
            if len(velocities) > 0:
                pose_list += [[path[0]]]
                velocity_list += [[velocities[0]]]
        all_poses = np.concatenate(pose_list, axis=0)
        all_velocities = np.concatenate(velocity_list, axis=0)

        return normalise_angles(all_poses), all_velocities

    def get_demo_path(self):
        path, velocities_WP = self.solve_ik()
        path_to_WP = path[:-1]
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=path[-1])
        path_to_trg, velocities_trg = self.solve_ik()
        return np.append(path_to_WP, path_to_trg, axis=0), \
               np.append(velocities_WP, velocities_trg, axis=0)


def setup_ROW_Env(seed, rank):
    env = ROWEnvInitialiser(seed, rank)
    ip = train_initial_policy(env)
    env.close()
    return ip
