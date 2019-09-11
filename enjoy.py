import argparse
import os

import numpy as np
import torch

from tqdm import tqdm

from envs.DRRewardEnvs import DRSparseEnv
# from envs.DRSparseEnv import DRSparseEnv
from envs.DishRackEnv import rack_lower, rack_upper
from envs.ReachOverWallEnv import ROWSparseEnv
from envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.utils import get_render_func


# workaround to unpickle olf model files
import sys

from im2state.utils import unnormalise_y

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
parser.add_argument('--i2s-load-dir', default='./trained_models/im2state/',
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
args = parser.parse_args()

args.det = not args.non_det
args.cuda = torch.cuda.is_available()


def main():
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # We need to use the same statistics for normalization as used in training
    policies = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"),
                          map_location=device)
    if args.e2e:
        e2e = policies
        e2e.eval()
        policies = None
    else:
        e2e = None

    estimator = torch.load(os.path.join(args.i2s_load_dir, args.image_layer + ".pt")) if \
        args.image_layer else None
    if estimator:
        estimator.eval()

    pose_estimator_info = (estimator, args.state_indices, rack_lower, rack_upper) if \
        args.image_layer else None

    env = make_vec_envs(DRSparseEnv, 'dish_rack', args.seed + 1000,
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

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() == "torso":
                torsoId = i

    i = 0
    total_successes = 0
    num_trials = 50
    low = torch.Tensor([-0.3] * 7)
    high = torch.Tensor([0.3] * 7)
    # init_rews = np.zeros((args.num_processes, 1))
    # rews = init_rews.copy()
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
        # obs, step_rews, dones, _ = env.step(action)
        # rews += step_rews.cpu().numpy()
        if np.all(dones):
            # print(rews)
            i += args.num_processes
            rew = sum([int(rew > 0) for rew in rews])
            total_successes += rew
            # rews = init_rews.copy()

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')

    p_succ = 100 * total_successes / i
    print(f"{p_succ}% successful")
    return p_succ


if __name__ == "__main__":
    # policy_names = [
    #     "50p_0_dish_rack",
    #     "50p_16_dish_rack",
    #     "50p_32_dish_rack",
    # ]
    policy_names = [
        # "50p_1cm_0_reach_over_wall_static",
        # "50p_1cm_16_reach_over_wall_static",
        # "50p_1cm_32_reach_over_wall_static",
        # "50p_2cm_0_reach_over_wall_static",
        # "50p_2cm_16_reach_over_wall_static",
        # "50p_2cm_32_reach_over_wall_static",
        # "50p_4cm_0_reach_over_wall_static",
        # "50p_4cm_16_reach_over_wall_static",
        # "50p_4cm_32_reach_over_wall_static",
        # "60p_1cm_0_reach_over_wall_static",
        # "60p_1cm_16_reach_over_wall_static",
        # "60p_1cm_32_reach_over_wall_static",
        # "60p_2cm_0_reach_over_wall_static",
        # "60p_2cm_16_reach_over_wall_static",
        # "60p_2cm_32_reach_over_wall_static",
        # "60p_4cm_0_reach_over_wall_static",
        # "60p_4cm_16_reach_over_wall_static",
        # "60p_4cm_32_reach_over_wall_static",
        # "70p_1cm_0_reach_over_wall_static",
        # "70p_1cm_16_reach_over_wall_static",
        # "70p_1cm_32_reach_over_wall_static",
        # "70p_2cm_0_reach_over_wall_static",
        # "70p_2cm_16_reach_over_wall_static",
        # "70p_2cm_32_reach_over_wall_static",
        # "70p_4cm_0_reach_over_wall_static",
        # "70p_4cm_16_reach_over_wall_static",
        # "70p_4cm_32_reach_over_wall_static",
        # "80p_1cm_0_reach_over_wall_static",
        # "80p_1cm_16_reach_over_wall_static",
        # "80p_1cm_32_reach_over_wall_static",
        # "80p_2cm_0_reach_over_wall_static",
        # "80p_2cm_16_reach_over_wall_static",
        # "80p_2cm_32_reach_over_wall_static",
        # "80p_4cm_0_reach_over_wall_static",
        # "80p_4cm_16_reach_over_wall_static",
        # "80p_4cm_32_reach_over_wall_static",
        # "90p_1cm_0_reach_over_wall_static",
        # "90p_1cm_16_reach_over_wall_static",
        # "90p_1cm_32_reach_over_wall_static",
        # "90p_2cm_0_reach_over_wall_static",
        # "90p_2cm_16_reach_over_wall_static",
        # "90p_2cm_32_reach_over_wall_static",
        # "90p_4cm_0_reach_over_wall_static",
        # "90p_4cm_16_reach_over_wall_static",
        # "90p_4cm_32_reach_over_wall_static",
        "100p_1cm_0_reach_over_wall_static",
        "100p_1cm_16_reach_over_wall_static",
        "100p_1cm_32_reach_over_wall_static",
        "100p_2cm_0_reach_over_wall_static",
        "100p_2cm_16_reach_over_wall_static",
        "100p_2cm_32_reach_over_wall_static",
        "100p_4cm_0_reach_over_wall_static",
        "100p_4cm_16_reach_over_wall_static",
        "100p_4cm_32_reach_over_wall_static",
    ]
    p_succs = []
    for policy in tqdm(policy_names):
        args.env_name = policy
        print(args.env_name)
        p_succs += [main()]
        torch.save(list(zip(policy_names, p_succs)), f"{args.save_as}_results.pt")
