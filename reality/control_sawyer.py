# The files in this directory can be used to run a trained policy on a real Sawyer arm. However it
# turned out the code had to be in python2. The python2 translation that was used is on the
# 'python2' branch.

import argparse
import os

import rospy
import torch
from baselines.common.vec_env import DummyVecEnv

from envs.ImageObsVecEnvWrapper import RealImageObsVecEnvWrapper
from envs.envs import wrap_initial_policies, VecPyTorch
from envs.wrappers import ClipActions, PoseEstimatorVecEnvWrapper
from reality.CameraConnection import CameraConnection
from reality.RealDishRackEnv import RealDishRackEnv


parser = argparse.ArgumentParser(description='Demonstrate Policy on Real Robot')
parser.add_argument('--policy-name', default=None,
                    help='trained policy to use')
parser.add_argument('--pose-est', default=None,
                    help='network taking images as input and giving state as output')
parser.add_argument('--load-dir', default='./trained_models/ppo/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--pe-load-dir', default='./trained_models/im2state/',
                    help='directory to save agent logs (default: ./trained_models/)')

parser.add_argument('--abs-to-rel', action='store_true', default=False)
parser.add_argument('--device-num', type=int, default=0, help='select CUDA device')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


def make_env_fn():
    def _thunk():
        env = RealDishRackEnv()
        env = ClipActions(env)

        return env

    return _thunk


def make_env(device, camera, policies, pose_estimator):
    env_fn = [make_env_fn()]
    vec_env = DummyVecEnv(env_fn)

    base_env = vec_env.envs[0]
    low = base_env.normalize_low
    high = base_env.normalize_high
    state_to_est = base_env.state_to_estimate

    vec_env = wrap_initial_policies(vec_env, device, policies)

    vec_env = RealImageObsVecEnvWrapper(vec_env, (128, 128), camera)

    vec_env = VecPyTorch(vec_env, device)

    vec_env = PoseEstimatorVecEnvWrapper(vec_env, device, pose_estimator, state_to_est,
                                         low, high, abs_to_rel=True)

    return vec_env


def main():
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.device_num}" if args.cuda else "cpu")

    policies = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"),
                          map_location=torch.device('cpu'))

    pose_estimator = torch.load(os.path.join(args.i2s_load_dir, args.image_layer + ".pt")) if \
        args.image_layer else None

    with CameraConnection((128, 128)) as camera:
        env = make_env(device, camera, policies, pose_estimator)
        null_action = torch.zeros((1, env.action_space.shape[0]))
        print("Executing policy on real robot. Press Ctrl-C to stop...")

        while not rospy.is_shutdown():
            env.step(null_action)

    print("Done.")


if __name__ == '__main__':
    main()
