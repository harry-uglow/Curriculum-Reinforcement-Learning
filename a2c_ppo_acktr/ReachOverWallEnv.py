import os
import torch

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.residual.ROW_utils import normalise_coords, normalise_angles, xl, xu, yu, yl, \
    train_initial_policy
from a2c_ppo_acktr.residual.initial_policy_model import InitialPolicy

from a2c_ppo_acktr.vrep_utils import check_for_errors, VrepEnv

np.set_printoptions(precision=2, linewidth=200)  # DEBUG

dir_path = os.getcwd()
scene_path = dir_path + '/reach_over_wall.ttt'

num_joints = 7


class ReachOverWallEnv(VrepEnv):

    observation_space = spaces.Box(np.array([0] * 11), np.array([1] * 11), dtype=np.float32)
    action_space = spaces.Box(np.array([-1] * num_joints), np.array([1] * num_joints),
                              dtype=np.float32)
    target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
    timestep = 0

    def __init__(self, seed, rank, initial_policy, ep_len=128, headless=True):
        super().__init__(rank, headless)

        self.target_pos = np.array([0.3, -0.5, 0.1])  # TODO: Obtain
        self.waypoint_pos = np.array([0, -0.5, 0.45])  # TODO: Obtain
        self.target_norm = normalise_coords(self.target_pos)
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed + rank)
        self.ep_len = ep_len
        self.initial_policy = initial_policy

        return_code = vrep.simxSynchronous(self.cid, enable=True)
        check_for_errors(return_code)

        return_code = vrep.simxLoadScene(self.cid, scene_path, 0, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

        # Get the initial configuration of the robot (needed to later reset the robot's pose)
        self.init_config_tree, _, _, _ = self.call_lua_function('get_configuration_tree')
        _, self.init_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = self.init_joint_angles

        self.joint_handles = np.array([None] * num_joints)
        for i in range(num_joints):
            return_code, handle = vrep.simxGetObjectHandle(self.cid, 'Sawyer_joint' + str(i + 1),
                                                           vrep.simx_opmode_blocking)
            check_for_errors(return_code)
            self.joint_handles[i] = handle

        return_code, self.end_handle = vrep.simxGetObjectHandle(self.cid,
                "Waypoint_tip", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        self.end_pose = self.get_end_pose()
        _, self.target_handle = vrep.simxGetObjectHandle(self.cid,
                "Cube", vrep.simx_opmode_blocking)
        _, self.wall_handle = vrep.simxGetObjectHandle(self.cid,
                "Wall", vrep.simx_opmode_blocking)
        self.init_wall_pos = vrep.simxGetObjectPosition(self.cid, self.wall_handle,
                -1, vrep.simx_opmode_blocking)[1]
        self.wall_distance = normalise_coords(self.init_wall_pos[0], lower=0, upper=1)
        self.init_wall_rot = vrep.simxGetObjectOrientation(self.cid,
                self.wall_handle, -1, vrep.simx_opmode_blocking)[1]
        self.wall_orientation = self.init_wall_rot

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

    def reset(self):
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree,
                               floats=self.init_joint_angles)
        vrep.simxSetObjectPosition(self.cid, self.wall_handle, -1, self.init_wall_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.wall_handle, -1, self.init_wall_rot,
                                      vrep.simx_opmode_blocking)

        self.target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.joint_angles = self.init_joint_angles
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        ip_input = torch.Tensor(normalise_angles(self.joint_angles)).reshape((1, num_joints))
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
        _, curr_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = np.array(curr_joint_angles)
        norm_joints = normalise_angles(self.joint_angles)
        self.end_pose = self.get_end_pose()

        return np.concatenate((norm_joints, self.target_norm, [self.wall_distance]))

    def update_sim(self):
        for handle, velocity in zip(self.joint_handles, self.target_velocities):
            return_code = vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot)
            check_for_errors(return_code)
        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1,
                                          vrep.simx_opmode_blocking)[1]
        return np.array(pose)


class ROWRandomTargetEnv(ReachOverWallEnv):

    def reset(self):
        self.target_pos[0] = self.np_random.uniform(xl, xu)
        self.target_pos[1] = self.np_random.uniform(yl, yu)
        self.target_norm = normalise_coords(self.target_pos)
        vrep.simxSetObjectPosition(self.cid, self.target_handle, -1, self.target_pos,
                                   vrep.simx_opmode_blocking)
        return super(ROWRandomTargetEnv, self).reset()


class ROWEnvInitialiser(ReachOverWallEnv):

    def __init__(self, seed, rank):
        ip = InitialPolicy(num_joints, num_joints)
        super().__init__(seed, rank, ip, headless=False) # DEBUG

    def solve_ik(self):
        end = self.get_end_pose()
        end_x = end[0]
        end_z = end[2]

        strings = ['IK_GroupW', 'tip_waypoint'] \
            if end_x < -0.005 and end_z < 0.445 - end_x \
            else ['IK_GroupT', 'tip_target']

        _, path, _, _ = self.call_lua_function('solve_ik', strings=strings)
        num_path_points = len(path) // num_joints
        path = np.reshape(path, (num_path_points, num_joints))
        distances = np.array([path[i + 1] - path[i]
                              for i in range(0, len(path) - 1)])
        velocities = distances * 20  # Distances should be covered in 0.05s
        return path, velocities

    # def get_initial_data(self, num_samples=5, scale=0.01):
    #     path, velocities = self.get_demo_path()
    #     poses = np.array([np.random.multivariate_normal(pose, scale * np.identity(num_joints),
    #                                                     num_samples) for pose in path])
    #     poses = np.reshape(poses, (len(path) * num_samples, num_joints))
    #
    #     self.poses = poses
    #
    #     return path, velocities

    def get_demo_path(self):
        path, velocities_WP = self.solve_ik()
        path_to_WP = path[:-1]
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=path[-1])
        path_to_trg, velocities_trg = self.solve_ik()
        return normalise_angles(np.append(path_to_WP, path_to_trg[:-1], axis=0)), \
               np.append(velocities_WP, velocities_trg, axis=0)


def setup_ROW_Env(seed, rank):
    env = ROWEnvInitialiser(seed, rank)
    ip = train_initial_policy(env)
    return ip
