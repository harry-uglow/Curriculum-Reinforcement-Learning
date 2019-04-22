import numpy as np
from gym import spaces
import vrep

from a2c_ppo_acktr.envs.VrepEnv import check_for_errors, VrepEnv


# Normalise coordinates so all are in range [0, 1].
def normalise_coords(coords, lower, upper):
    return (coords - lower) / (upper - lower)


# Normalise joint angles so -pi -> 0, 0 -> 0.5 and pi -> 1. (mod pi)
def normalise_angles(angles):
    js = angles / np.pi
    rem = lambda x: x - x.astype(int)
    return np.array([rem((j + (np.abs(j) // 2 + 1.5) * 2) / 2.) for j in js])


class SawyerEnv(VrepEnv):
    num_joints = 7
    action_space = spaces.Box(np.array([-1] * num_joints), np.array([1] * num_joints),
                              dtype=np.float32)
    target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])

    def __init__(self, seed, rank, scene_path, headless):
        super().__init__(rank, headless)

        self.np_random = np.random.RandomState()
        self.np_random.seed(seed + rank)

        return_code = vrep.simxSynchronous(self.cid, enable=True)
        check_for_errors(return_code)

        return_code = vrep.simxLoadScene(self.cid, scene_path, 0, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

        # Get the initial configuration of the robot (needed to later reset the robot's pose)
        self.init_config_tree, _, _, _ = self.call_lua_function('get_configuration_tree')
        _, self.init_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = self.init_joint_angles

        self.joint_handles = np.array([None] * self.num_joints)
        for i in range(self.num_joints):
            return_code, handle = vrep.simxGetObjectHandle(self.cid, 'Sawyer_joint' + str(i + 1),
                                                           vrep.simx_opmode_blocking)
            check_for_errors(return_code)
            self.joint_handles[i] = handle

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

    def reset(self):
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree,
                               floats=self.init_joint_angles)
        self.joint_angles = self.init_joint_angles
        self.target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])

    def _get_obs(self):
        _, curr_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = np.array(curr_joint_angles)
        norm_joints = normalise_angles(self.joint_angles)

        return norm_joints

    def update_sim(self):
        for handle, velocity in zip(self.joint_handles, self.target_velocities):
            return_code = vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot)
            check_for_errors(return_code)
        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)
