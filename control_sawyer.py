from __future__ import with_statement
from __future__ import absolute_import
import argparse
import os

import numpy as np
import rospy
import torch
from baselines.common.vec_env import DummyVecEnv

from envs.ImageObsVecEnvWrapper import ImageObsVecEnvWrapper, SimImageObsVecEnvWrapper
from envs.envs import wrap_initial_policies, VecPyTorch
from envs.wrappers import ClipActions, PoseEstimatorVecEnvWrapper, ReferenceEnv
from im2state.utils import unnormalise_y
from reality.CameraConnection import CameraConnection
from reality.RealDishRackEnv import RealDishRackEnv


parser = argparse.ArgumentParser(description=u'Demonstrate Policy on Real Robot')
parser.add_argument(u'--policy-name', default=None,
                    help=u'trained policy to use')
parser.add_argument(u'--pose-est', default=None,
                    help=u'network taking images as input and giving state as output')
parser.add_argument(u'--load-dir', default=u'./trained_models/im2state/',
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
        # env = ReferenceEnv(env)
        env = ClipActions(env)

        return env

    return _thunk


def make_env(device, camera):
    env_fn = [make_env_fn(camera, [128, 128])]
    vec_env = DummyVecEnv(env_fn)

    vec_env = VecPyTorch(vec_env, device)

    return vec_env


def main():
    torch.set_num_threads(1)
    device = torch.device(u"cuda:" + args.device_num if args.cuda else u"cpu")

    e2e = torch.load(os.path.join(args.load_dir, args.policy_name + ".pt"),
                          map_location=device)

    low = torch.Tensor([-0.3] * 7)
    high = torch.Tensor([0.3] * 7)

    #camera = None
    #if True:
    with CameraConnection([128, 128]) as camera:
        env = make_env(device, camera)
        print u"Executing policy on real robot. Press Ctrl-C to stop..."

        obs = env.reset()
        while not rospy.is_shutdown():
            images = torch.Tensor(np.transpose(env.get_images(), (0, 3, 1, 2))).to(device)
            output = e2e.predict(images, obs[:, :7])
            action = unnormalise_y(output, low, high)
            obs = env.step(action)[0]

    print u"Done."


if __name__ == u'__main__':
    main()
