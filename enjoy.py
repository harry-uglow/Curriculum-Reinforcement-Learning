import argparse
import os

import torch

from a2c_ppo_acktr.envs.ResidualVecEnvWrapper import get_residual_layers
from a2c_ppo_acktr.envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.utils import get_render_func


# workaround to unpickle olf model files
import sys

from im2state.utils import format_images, unnormalise_y

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--i2s-load-dir', default='./trained_models/im2state/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--image-layer', default=None,
                    help='network taking images as input and giving state as output')
parser.add_argument('--state-indices', nargs='+', type=int)
args = parser.parse_args()

args.det = not args.non_det

# We need to use the same statistics for normalization as used in training
policies = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

im2state = torch.load(os.path.join(args.i2s_load_dir, args.image_layer + ".pt")) if \
    args.image_layer else None
if im2state:
    im2state.eval()

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None, args.add_timestep, 'cpu',
                    False, policies, show=True, no_norm=True)
null_action = torch.zeros((1, env.action_space.shape[0]))
low = env.observation_space.low[args.state_indices]
high = env.observation_space.high[args.state_indices]

policy_wrappers = get_residual_layers(env)

# Get a render function
render_func = get_render_func(env)

obs = env.reset()

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:

    with torch.no_grad():
        if im2state:
            image = torch.from_numpy(format_images(env.get_images())).float()
            part_state = unnormalise_y(im2state(image).numpy(), low, high)
            obs = obs.numpy()
            obs[:, args.state_indices] = part_state
            for policy in policy_wrappers:
                policy.curr_obs = policy.normalize_obs(obs)

    # Obser reward and next obs
    obs, _, done, _ = env.step(null_action)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
