import argparse
import os

import torch

from a2c_ppo_acktr.envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.utils import get_render_func


# workaround to unpickle olf model files
import sys
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
actor_critic, ob_rms, initial_policies = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

im2state = torch.load(os.path.join(args.load_dir, args.image_layer + ".pt")) if \
    args.image_layer is not None else None
im2state.eval()

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None, args.add_timestep, 'cpu',
                    False, initial_policies, vis=True)

# Get a render function
render_func = get_render_func(env)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')
    image = render_func('rgb_array')


if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:

    with torch.no_grad():
        if im2state:
            part_state = im2state(image)
            obs[:, args.state_indices] = part_state
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, _, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
        image = render_func('rgb_array')
