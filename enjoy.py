import argparse
import os

import numpy as np
import torch

from envs.DishRackEnv import rack_lower, rack_upper
from envs.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_render_func
from envs.pipelines import pipelines

import sys

from pose_estimator.utils import unnormalise_y

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training CPU processes to use (default: 16)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--save-as', default='test',
                    help='where to save % success results')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--pe-load-dir', default='./trained_models/pe/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--image-layer', default=None,
                    help='network taking images as input and giving state as output')
parser.add_argument('--state-indices', nargs='+', type=int)
parser.add_argument('--rip', action='store_true', default=False)
parser.add_argument('--e2e', action='store_true', default=False)
parser.add_argument('--pipeline', default=None, help='Task pipeline the policy was trained on')
args = parser.parse_args()

args.det = not args.non_det
args.cuda = torch.cuda.is_available()


def main():
    device = torch.device("cuda:0" if args.cuda else "cpu")
    policies = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"),
                          map_location=device)
    if args.e2e:
        e2e = policies
        e2e.eval()
        policies = None
    else:
        e2e = None

    estimator = torch.load(os.path.join(args.pe_load_dir, args.image_layer + ".pt")) if \
        args.image_layer else None
    if estimator:
        estimator.eval()

    pose_estimator_info = (estimator, args.state_indices, rack_lower, rack_upper) if \
        args.image_layer else None

    pipeline = pipelines[args.pipeline]

    env = make_vec_envs(pipeline['sparse'], pipeline['task'], args.seed + 1000,
                        args.num_processes, None, None, device, False, policies,
                        show=(args.num_processes == 1), no_norm=True,
                        pose_estimator=pose_estimator_info)
    null_action = torch.zeros((1, env.action_space.shape[0]))

    # Get a render function
    render_func = get_render_func(env)

    if e2e:
        env.get_images(mode='activate')
    obs = env.reset()

    if render_func is not None:
        render_func('human')

    i = 0
    total_successes = 0
    num_trials = 50
    low = torch.Tensor([-0.3] * 7)
    high = torch.Tensor([0.3] * 7)
    while i < num_trials:
        with torch.no_grad():
            if e2e:
                images = torch.Tensor(np.transpose(env.get_images(), (0, 3, 1, 2))).to(device)
                output = e2e.predict(images, obs[:, :7])
                action = unnormalise_y(output, low, high)
            else:
                action = null_action

        # Obser reward and next obs
        obs, rews, dones, _ = env.step(action)
        if np.all(dones):
            i += args.num_processes
            rew = sum([int(rew > 0) for rew in rews])
            total_successes += rew

        if render_func is not None:
            render_func('human')

    p_succ = 100 * total_successes / i
    print(f"{p_succ}% successful")


if __name__ == "__main__":
    main()
