import numpy as np
import torch
from baselines.common.vec_env import VecEnvWrapper
from gym import spaces, ActionWrapper

from envs.ImageObsVecEnvWrapper import get_image_obs_wrapper
from envs.ResidualVecEnvWrapper import get_residual_layers
from im2state.utils import unnormalise_y


class PoseEstimatorVecEnvWrapper(VecEnvWrapper):
    """
    Uses a pose estimator to estimate the state from the image. Wrapping this environment
    around a ResidualVecEnvWrapper makes it possible to use a full state policy on an environment
    with images as observations
    """
    def __init__(self, venv, device, pose_estimator, state_to_estimate, low, high,
                 abs_to_rel=False):
        super().__init__(venv)
        self.image_obs_wrapper = get_image_obs_wrapper(venv)
        assert self.image_obs_wrapper is not None
        self.estimator = pose_estimator.to(device)
        self.estimator.eval()
        self.policy_layers = get_residual_layers(venv)
        self.state_obs_space = self.policy_layers[0].observation_space
        self.state_to_estimate = state_to_estimate
        self.state_to_use = [i for i in range(self.state_obs_space.shape[0])
                             if i not in state_to_estimate]
        self.low = low
        self.high = high
        self.curr_image = None
        self.abs_to_rel = abs_to_rel
        self.target_z = np.array(self.get_images(mode="target_height"))
        self.junk = None
        self.abs_estimations = None

    def step_async(self, actions):
        with torch.no_grad():
            net_output = self.estimator.predict(self.curr_image).cpu().numpy()
            estimation = net_output if self.low is None else unnormalise_y(net_output,
                                                                           self.low, self.high)
            if self.abs_estimations is None:
                self.abs_estimations = np.array([estimation])
            else:
                self.abs_estimations = np.append(self.abs_estimations, [estimation], axis=0)

            obs = np.zeros((self.num_envs, *self.state_obs_space.shape))
            estimation = np.median(self.abs_estimations, axis=0)
            obs[:, self.state_to_use] = self.image_obs_wrapper.curr_state_obs[:, self.state_to_use]
            if self.abs_to_rel:
                full_pos_estimation = np.append(estimation[:, :2], self.target_z, axis=1)
                # rack_to_trg = self.base_env.get_position(self.base_env.target_handle) - \
                #               self.base_env.get_position(self.base_env.rack_handle)
                # full_pos_estimation += rack_to_trg
                actual_plate_pos = np.array(self.get_images(mode='plate'))
                relative_estimation = full_pos_estimation - actual_plate_pos
                estimation = np.append(relative_estimation, estimation[:, 2:], axis=1)
            # FOR ADJUSTING FROM RACK ESTIMATION TO TARGET OBSERVATIONS
            # rack_to_trg = self.base_env.get_position(self.base_env.target_handle) - \
            #               self.base_env.get_position(self.base_env.rack_handle)
            # estimation[:, :-1] += rack_to_trg[:-1]
            obs[:, self.state_to_estimate] = estimation
            for policy in self.policy_layers:
                policy.curr_obs = obs
        self.venv.step_async(actions)

    def step_wait(self):
        self.curr_image, rew, done, info = self.venv.step_wait()
        if np.all(done):
            self.abs_estimations = None
        return self.curr_image, rew, done, info

    def reset(self):
        self.curr_image = self.venv.reset()
        return self.curr_image


class ClipActions(ActionWrapper):
    def __init__(self, env):
        super(ClipActions, self).__init__(env)

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


# TODO: Scale properly to support boundaries where low != -high
class BoundPositionVelocity(ActionWrapper):
    def __init__(self, env):
        super(BoundPositionVelocity, self).__init__(env)

    def action(self, action):
        pos = action[:3]

        if not ((pos >= self.action_space.low[0]) & (pos <= self.action_space.high[0])).all():
            pos /= np.max(np.abs(pos))
            pos *= self.action_space.high[0]

        return action


class ScaleActions(ActionWrapper):
    def __init__(self, env, factor):
        self.factor = factor
        super(ScaleActions, self).__init__(env)

    def action(self, action):
        action *= self.factor
        return action


class E2EVecEnvWrapper(VecEnvWrapper):
    """
    Used to train an end-to-end policy. Not useful in this project, training unfeasibly long.
    """
    def __init__(self, venv):
        res = venv.get_images(mode='activate')[0]
        image_obs_space = spaces.Box(0, 255, [3, *res], dtype=np.uint8)
        state_obs_space = venv.observation_space
        observation_space = spaces.Tuple((image_obs_space, state_obs_space))
        observation_space.shape = (image_obs_space.shape, state_obs_space.shape)
        super().__init__(venv, observation_space)
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


class InitialController(ActionWrapper):
    """
    This environment wrapper moves the subject directly toward the target position/orientation at
    every step. It can be used to initialise learning without the need for reward shaping.
    """
    def __init__(self, env):
        super(InitialController, self).__init__(env)
        self.base_env = env.unwrapped

    def action(self, action):
        vec = self.base_env.target_pos - self.base_env.subject_pos
        if not ((vec >= self.action_space.low[0]) & (vec <= self.action_space.high[0])).all():
            vec /= np.max(np.abs(vec))
            vec *= self.action_space.high[0]

        full_vec = vec + action[:3]

        rot = action[3:]

        full_action = np.append(full_vec, rot)

        return full_action
