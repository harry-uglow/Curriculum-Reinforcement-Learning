import numpy as np
from gym import spaces
import vrep

from envs.VrepEnv import catch_errors, VrepEnv


class SawyerEnv(VrepEnv):
    num_joints = 7
    action_space = spaces.Box(np.array([-0.1] * 5), np.array([0.1] * 5), dtype=np.float32)
    target_point = np.array([0.] * 5)
    scale = 0.01
    identity = scale * np.identity(num_joints)

    def __init__(self, *args, random_joints=True):
        super().__init__(*args)

        self.random_joints = random_joints
        self.np_random = np.random.RandomState()

        # Get the initial configuration of the robot (needed to later reset the robot's pose)
        self.init_config_tree, _, _, _ = self.call_lua_function('get_configuration_tree')
        _, self.init_joint_angles, _, _ = self.call_lua_function('get_joint_angles')

        self.joint_handles = np.array([None] * self.num_joints)
        for i in range(self.num_joints):
            handle = catch_errors(vrep.simxGetObjectHandle(self.cid, 'Sawyer_joint' + str(i + 1),
                                                           vrep.simx_opmode_blocking))
            self.joint_handles[i] = handle
        self.mv_target = catch_errors(vrep.simxGetObjectHandle(self.cid, 'MvTarget',
                                                               vrep.simx_opmode_blocking))

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        catch_errors(vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking))

    def seed(self, seed=None):
        self.np_random.seed(seed)

    def reset(self):
        if self.random_joints:
            initial_pose = self.np_random.multivariate_normal(self.init_joint_angles, self.identity)
        else:
            initial_pose = self.init_joint_angles
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=initial_pose)
        self.target_point = np.array([0.] * 5)

    def _get_obs(self):
        _, joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        assert len(joint_angles) == self.num_joints

        return joint_angles

    def update_sim(self):
        self.call_lua_function('update_robot_movement', floats=self.target_point * 0.05)

        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)

    def compute_target_pose(self):
        _, target_pose, _, _ = self.call_lua_function('solve_ik')
        return np.array(target_pose)
