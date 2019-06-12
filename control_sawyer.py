from __future__ import with_statement
from __future__ import absolute_import
import argparse
import os

import rospy
import torch
from baselines.common.vec_env import DummyVecEnv

from envs.ImageObsVecEnvWrapper import ImageObsVecEnvWrapper, SimImageObsVecEnvWrapper
from envs.envs import wrap_initial_policies, VecPyTorch
from envs.wrappers import ClipActions, PoseEstimatorVecEnvWrapper, ReferenceEnv
from reality.CameraConnection import CameraConnection
from reality.RealDishRackEnv import RealDishRackEnv


parser = argparse.ArgumentParser(description=u'Demonstrate Policy on Real Robot')
parser.add_argument(u'--policy-name', default=None,
                    help=u'trained policy to use')
parser.add_argument(u'--pose-est', default=None,
                    help=u'network taking images as input and giving state as output')
parser.add_argument(u'--load-dir', default=u'./trained_models/ppo/',
                    help=u'directory to save agent logs (default: ./trained_models/)')
parser.add_argument(u'--pe-load-dir', default=u'./trained_models/im2state/',
                    help=u'directory to save agent logs (default: ./trained_models/)')

parser.add_argument(u'--abs-to-rel', action=u'store_true', default=False)
parser.add_argument(u'--device-num', default='0', help=u'select CUDA device')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()


def make_env_fn(*args):
    def _thunk():
        env = RealDishRackEnv(*args)
        env = ReferenceEnv(env)
        env = ClipActions(env)

        return env

    return _thunk


def make_env(device, camera, policies, pose_estimator):
    env_fn = [make_env_fn(camera, [128, 128])]
    vec_env = DummyVecEnv(env_fn)

    base_env = vec_env.envs[0]
    low = base_env.normalize_low
    high = base_env.normalize_high
    state_to_est = base_env.state_to_estimate

    vec_env = wrap_initial_policies(vec_env, device, policies)

    vec_env = SimImageObsVecEnvWrapper(vec_env)

    vec_env = VecPyTorch(vec_env, device)

    vec_env = PoseEstimatorVecEnvWrapper(vec_env, device, pose_estimator, state_to_est,
                                         low, high, abs_to_rel=True)

    return vec_env


def main():
    torch.set_num_threads(1)
    device = torch.device(u"cuda:" + args.device_num if args.cuda else u"cpu")

    policies = torch.load(os.path.join(args.load_dir, args.policy_name + u".pt"),
                          map_location=torch.device(u'cpu'))

    pose_estimator = torch.load(os.path.join(args.pe_load_dir, args.pose_est + u".pt")) if \
        args.pose_est else None

    #camera = None
    #if True:
    with CameraConnection([128, 128]) as camera:
        env = make_env(device, camera, policies, pose_estimator)
        null_action = torch.zeros((1, env.action_space.shape[0]))
        print u"Executing policy on real robot. Press Ctrl-C to stop..."

        env.reset()
        while not rospy.is_shutdown():
            env.step(null_action)

    print u"Done."


if __name__ == u'__main__':
    main()
