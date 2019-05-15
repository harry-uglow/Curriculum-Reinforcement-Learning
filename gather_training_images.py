import argparse
import copy
import math
import os
import time

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs.envs import make_vec_envs, get_vec_normalize
from im2state.model import CNN

from tqdm import tqdm

from im2state.utils import format_images, normalise_coords

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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir,
                         args.add_timestep, device, False, policies, no_norm=True)

    null_action = torch.zeros((args.num_processes, envs.action_space.shape[0]))

    obs = envs.reset()
    image = format_images(envs.get_images())
    mask = format_images(envs.get_images(mode='mask'))

    images = np.zeros((args.num_steps, *image.shape[1:]), dtype=np.uint8)
    images[0: args.num_processes] = image
    masks = np.zeros((args.num_steps, *mask.shape[1:]), dtype=np.uint8)
    masks[0: args.num_processes] = mask
    positions = np.zeros((args.num_steps, len(args.state_indices)))
    positions[0: args.num_processes] = obs[:, args.state_indices]

    for i in tqdm(range(1, args.num_steps // args.num_processes)):

        obs, _, done, _ = envs.step(null_action)
        start_index = args.num_processes * i
        positions[start_index:start_index + args.num_processes] = obs[:, args.state_indices]

        img = format_images(envs.get_images())
        mask = format_images(envs.get_images(mode='mask'))
        images[start_index:start_index + args.num_processes] = img
        masks[start_index:start_index + args.num_processes] = mask

    envs.close()

    save_path = os.path.join('training_data')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    low = envs.observation_space.low[args.state_indices]
    high = envs.observation_space.high[args.state_indices]
    res = images.shape[2]

    torch.save([images, masks, positions, low, high],
               os.path.join(save_path, f'{args.env_name}_{res}_{args.num_steps}_masks.pt'))


if __name__ == "__main__":
    main()
