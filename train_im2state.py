import argparse
import copy
import os
import time

import numpy as np
import torch

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.utils import get_render_func
from a2c_ppo_acktr.visualize import visdom_plot
from im2state.model import CNN

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    policies = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

    # CREATE ENVIRONMENT
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir,
                         args.add_timestep, device, False, policies, no_norm=True)

    null_action = torch.zeros((args.num_processes, envs.action_space.shape[0]))
    # Get a render function
    render_func = get_render_func(envs)
    if render_func is None:
        raise AssertionError("Environment has no render function")

    net = CNN(envs.observation_space.shape[0], envs.action_space.shape[0])
    net.to(device)

    num_training_images = 2048

    # start = time.time() # MAYBE USE TIMINGS
    # start_update = start

    image = render_func('rgb_array').transpose(2, 0, 1)

    images = np.zeros((num_training_images, *image.shape), dtype=np.uint8)
    images[0] = image
    positions = np.zeros((num_training_images, len(args.state_indices)))

    obs = envs.reset()
    positions[0] = obs[:, args.state_indices]

    for i in range(1, num_training_images):

        obs, _, done, _ = envs.step(null_action)
        positions[i] = obs[:, args.state_indices]

        if render_func is not None:
            images[i] = render_func('rgb_array').transpose(2, 0, 1)

    # TRAIN NN

    # save for every interval-th episode or for the last epoch
    if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        # TODO: Save better and include initial policy
        save_model = net
        if args.cuda:
            save_model = copy.deepcopy(net).cpu()

        torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

    if j % args.log_interval == 0:
        end = time.time()
        print("Training epoch .... ")
        print("Update length: ", end - start_update)
        start_update = end

    if args.vis and j % args.vis_interval == 0:
        # PLOT IN SOME WAY
        pass
        # try:
        #     # visdom_plot(args.log_dir, args.env_name, args.algo, args.num_env_steps)
        # except IOError:
        #     pass


if __name__ == "__main__":
    main()
