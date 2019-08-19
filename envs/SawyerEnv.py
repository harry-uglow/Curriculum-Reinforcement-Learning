import numpy as np
from gym import spaces
import vrep

from envs.VrepEnv import catch_errors, VrepEnv


class SawyerEnv(VrepEnv):
    num_joints = 7
    action_space = spaces.Box(np.array([-0.1, -0.1, -0.1, -0.3, -0.3, -0.3]),
                              np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]),
                              dtype=np.float32)
    target_velocity = np.array([0., 0., 0., 0., 0., 0.])
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
        self.target_velocity = np.array([0., 0., 0.])
        vrep.simxSetObjectPosition(self.cid, self.mv_target, vrep.sim_handle_parent,
                                   [0., 0., 0.], vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.mv_target, vrep.sim_handle_parent,
                                      [0., 0., 0.], vrep.simx_opmode_blocking)

    def _get_obs(self):
        _, joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        assert len(joint_angles) == self.num_joints

        return joint_angles

    def update_sim(self):
        vrep.simxSetObjectPosition(self.cid, self.mv_target, vrep.sim_handle_parent,
                                   [0., 0., 0.], vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.mv_target, vrep.sim_handle_parent,
                                      [0., 0., 0.], vrep.simx_opmode_blocking)
        target_pos = catch_errors(vrep.simxGetObjectPosition(self.cid, self.mv_target, -1,
                                                             vrep.simx_opmode_blocking))
        target_pos += self.target_velocity[:3] * 0.05
        vrep.simxSetObjectPosition(self.cid, self.mv_target, -1, target_pos,
                                   vrep.simx_opmode_blocking)
        target_rot = catch_errors(vrep.simxGetObjectOrientation(self.cid, self.mv_target, -1,
                                                                vrep.simx_opmode_blocking))
        target_rot += self.target_velocity[3:] * 0.05
        vrep.simxSetObjectOrientation(self.cid, self.mv_target, -1, target_rot,
                                      vrep.simx_opmode_blocking)
        target_pose = self.compute_target_pose()

        for handle, velocity in zip(self.joint_handles, target_pose):
            catch_errors(vrep.simxSetJointTargetPosition(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot))
        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)

    def compute_target_pose(self):
        _, target_pose, _, _ = self.call_lua_function('solve_ik')
        return np.array(target_pose)
