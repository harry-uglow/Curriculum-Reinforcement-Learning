from __future__ import with_statement
from __future__ import absolute_import
import numpy as np
import torch
from baselines.common.vec_env import VecEnvWrapper
from gym import spaces, ActionWrapper, Wrapper

from envs.ImageObsVecEnvWrapper import get_image_obs_wrapper
from envs.ResidualVecEnvWrapper import get_residual_layers
from envs.DishRackEnv import DishRackEnv
from im2state.utils import unnormalise_y


class PoseEstimatorVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, device, pose_estimator, state_to_estimate, low, high,
                 abs_to_rel=False):
        super(PoseEstimatorVecEnvWrapper, self).__init__(venv)
        self.image_obs_wrapper = get_image_obs_wrapper(venv)
        assert self.image_obs_wrapper is not None
        self.estimator = pose_estimator.to(device)
        self.estimator.eval()
        self.policy_layers = get_residual_layers(venv)
        self.state_obs_space = self.policy_layers[0].observation_space
        self.state_to_estimate = state_to_estimate
        self.state_to_use = [i for i in xrange(self.state_obs_space.shape[0])
                             if i not in state_to_estimate]
        self.low = low
        self.high = high
        self.curr_image = None
        self.abs_to_rel = abs_to_rel
        self.target_z = np.array(self.get_images(mode=u"target_height"))
        self.junk = None
        # self.abs_estimations = SORTED LISTS --- USE MEDIAN ESTIMATION

    def step_async(self, actions):
        with torch.no_grad():
            net_output = self.estimator.predict(self.curr_image).cpu().numpy()
            estimation = net_output if self.low is None else unnormalise_y(net_output,
                                                                           self.low, self.high)
            # self.abs_estimations
            print(estimation[0, 0])
            obs = np.zeros([self.num_envs] + list(self.state_obs_space.shape))
            obs[:, self.state_to_use] = self.image_obs_wrapper.curr_state_obs[:, self.state_to_use]
            print estimation
            if self.abs_to_rel:
                full_pos_estimation = np.append(estimation[:, :2], self.target_z, axis=1)
                print full_pos_estimation
                # rack_to_trg = self.base_env.get_position(self.base_env.target_handle) - \
                #               self.base_env.get_position(self.base_env.rack_handle)
                # full_pos_estimation += rack_to_trg
                actual_plate_pos = np.array(self.get_images(mode=u'plate'))
                print actual_plate_pos
                relative_estimation = full_pos_estimation - actual_plate_pos
                estimation = np.append(relative_estimation, estimation[:, 2:], axis=1)
            # FOR ADJUSTING FROM RACK ESTIMATION TO TARGET OBSERVATIONS
            # rack_to_trg = self.base_env.get_position(self.base_env.target_handle) - \
            #               self.base_env.get_position(self.base_env.rack_handle)
            # estimation[:, :-1] += rack_to_trg[:-1]
            print estimation
            obs[:, self.state_to_estimate] = estimation
            for policy in self.policy_layers:
                policy.curr_obs = obs
        self.venv.step_async(actions)

    def step_wait(self):
        self.curr_image, rew, done, info = self.venv.step_wait()
        return self.curr_image, rew, done, info

    def reset(self):
        self.curr_image = self.venv.reset()
        self.abs_estimations = np.array([])
        return self.curr_image


class ClipActions(ActionWrapper):
    def __init__(self, env):
        super(ClipActions, self).__init__(env)

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

class ReferenceEnv(Wrapper):
    def __init__(self, env):
        super(ReferenceEnv, self).__init__(env)
        print "Starting vrep"
        self.ref_env = DishRackEnv('dish_rack', 0, False)
        print "vrep started"

    def render(self, mode):
        if mode == 'plate' or mode == 'target_height':
            return self.ref_env.render(mode)
        else:
            return self.env.render(mode)

    def step(self, action):
        ob, rew, done, info = super(ReferenceEnv, self).step(action)
        self.ref_env.call_lua_function('set_joint_angles', ints=self.ref_env.init_config_tree, 
                                       floats=ob[:7]) 
        return ob, rew, done, info

    def reset(self):
        ob = super(ReferenceEnv, self).reset()
        self.ref_env.call_lua_function('set_joint_angles', ints=self.ref_env.init_config_tree, 
                                       floats=ob[:7]) 
        return ob



class E2EVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        res = venv.get_images(mode=u'activate')[0]
        image_obs_space = spaces.Box(0, 255, [3] + res, dtype=np.uint8)
        state_obs_space = venv.observation_space
        observation_space = spaces.Tuple((image_obs_space, state_obs_space))
        observation_space.shape = (image_obs_space.shape, state_obs_space.shape)
        super(E2EVecEnvWrapper, self).__init__(venv, observation_space)
        self.curr_state_obs = None
        self.last_4_image_obs = None

    def reset(self):
        self.curr_state_obs = self.venv.reset()
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return image_obs, self.curr_state_obs

    # Swap out state for image
    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        self.curr_state_obs = obs
        image_obs = np.transpose(self.venv.get_images(), (0, 3, 1, 2))
        return (image_obs, self.curr_state_obs), rew, done, info
