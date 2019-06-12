from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import numpy as np
import torch
from baselines.common.vec_env import VecEnvWrapper


def get_residual_layers(venv):
    if isinstance(venv, ResidualVecEnvWrapper):
        return [venv] + get_residual_layers(venv.venv)
    elif hasattr(venv, u'venv'):
        return get_residual_layers(venv.venv)
    return []


class ResidualVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, initial_policy, ob_rms, device, clipob=10., epsilon=1e-8):
        super(ResidualVecEnvWrapper, self).__init__(venv)
        self.ip = initial_policy
        self.ip.eval()
        self.ip.to(device)
        self.ob_rms = ob_rms
        self.ob_size = len(ob_rms.mean) if ob_rms else venv.observation_space.shape
        self.device = device
        self.clipob = clipob
        self.epsilon = epsilon
        self.curr_obs = None

    def normalize_obs(self, obs):
        if self.ob_rms:
            obs = obs[:, :self.ob_size]
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return torch.from_numpy(obs).float().to(self.device)
        else:
            return obs

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()

        self.curr_obs = obs
        return obs, rew, done, info

    def step_async(self, action):
        with torch.no_grad():
            _, ip_action, _, _ = self.ip.act(self.normalize_obs(self.curr_obs), None, None,
                                             deterministic=True)
        if self.ob_rms:
            ip_action = ip_action.squeeze(1).cpu().numpy()
        whole_action = ip_action + action
        self.venv.step_async(whole_action)

    def reset(self):
        obs = self.venv.reset()
        self.curr_obs = obs
        return obs
