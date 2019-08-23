import numpy as np
from gym import spaces
import vrep
from envs.SawyerEnv import SawyerEnv

from envs.VrepEnv import catch_errors, VrepEnv


class GoalDrivenEnv(SawyerEnv):
    # Cartesian control - orientation constraints don't matter
    action_space = spaces.Box(np.array([-0.02]*3 + [-1.]*3), np.array([0.02]*3 + [1.]*3),
                              dtype=np.float32)
    curr_action = np.array([0.] * 6)
    timestep = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subject_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Subject",
                                                                    vrep.simx_opmode_blocking))
        self.target_handle = catch_errors(vrep.simxGetObjectHandle(self.cid, "Target",
                                                                   vrep.simx_opmode_blocking))
        self.subject_pos = [0.]*3
        self.target_pos = [0.]*3

    def reset(self):
        super(GoalDrivenEnv, self).reset()
        self.timestep = 0
        self.curr_action = np.array([0.] * 6)

    def _get_obs(self):
        joint_obs = super(GoalDrivenEnv, self)._get_obs()
        self.target_pos = self.get_position(self.target_handle)
        self.subject_pos = self.get_position(self.subject_handle)
        pos_vector = self.target_pos - self.subject_pos

        return np.append(joint_obs, pos_vector)

    def update_sim(self):
        self.call_lua_function('update_robot_movement', floats=self.curr_action)

        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)
