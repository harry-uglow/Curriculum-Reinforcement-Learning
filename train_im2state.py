import argparse
import copy
import math
import os
import time

import numpy as np
import torch
from torch import nn, optim

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.utils import get_render_func
from a2c_ppo_acktr.visualize import visdom_plot
from im2state.model import CNN

from tqdm import tqdm

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    policies = torch.load(os.path.join(args.load_dir, args.initial_policy + ".pt"))

    # CREATE ENVIRONMENT
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir,
                         args.add_timestep, device, False, policies, no_norm=True)

    null_action = torch.zeros((args.num_processes, envs.action_space.shape[0]))
    # Get a render function
    render_func = get_render_func(envs)
    if render_func is None:
        raise AssertionError("Environment has no render function")

    net = CNN(3, len(args.state_indices))
    net.to(device)

    num_training_images = 2048

    # start = time.time() # MAYBE USE TIMINGS
    # start_update = start

    image = render_func('rgb_array').transpose(2, 0, 1)

    images = np.zeros((num_training_images, *image.shape), dtype=np.uint8)
    images[0: args.num_processes] = image
    positions = np.zeros((num_training_images, len(args.state_indices)))

    obs = envs.reset()
    positions[0: args.num_processes] = obs[:, args.state_indices]

    for i in tqdm(range(1, num_training_images // args.num_processes)):

        obs, _, done, _ = envs.step(null_action)
        start_index = args.num_processes * i
        positions[start_index:start_index + args.num_processes] = obs[:, args.state_indices]

        if render_func is not None:
            img = render_func('rgb_array').transpose(2, 0, 1)
            images[start_index:start_index + args.num_processes] = img

    # TRAIN NN
    save_path = os.path.join(args.save_dir, args.algo)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    optimizer = optim.SGD(net.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    num_test_examples = len(images) // 10

    train_x = torch.Tensor(images[num_test_examples:])
    train_y = torch.Tensor(positions[num_test_examples:])
    test_x = torch.Tensor(images[:num_test_examples])
    test_y = torch.Tensor(positions[:num_test_examples])

    min_test_loss = math.inf

    updates_with_no_improvement = 0

    # run the main training loop
    epochs = 0
    while updates_with_no_improvement < 5:
        epochs += 1
        actual_y = net(train_x)
        loss = criterion(actual_y, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss = criterion(net(test_x), test_y).item()
        if test_loss < min_test_loss:
            updates_with_no_improvement = 0
            min_test_loss = test_loss
        else:
            if updates_with_no_improvement == 0:
                save_model = net
                if args.cuda:
                    save_model = copy.deepcopy(net).cpu()

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            updates_with_no_improvement += 1

        if epochs % args.log_interval == 0:
            print(f"Training epoch {epochs} - validation loss: {test_loss}")

    # if args.vis and epochs % args.vis_interval == 0:
    #     # PLOT IN SOME WAY
    #     pass
    #     # try:
    #     #     # visdom_plot(args.log_dir, args.env_name, args.algo, args.num_env_steps)
    #     # except IOError:
    #     #     pass


if __name__ == "__main__":
    main()
