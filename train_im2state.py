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

    images, positions = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

    save_path = os.path.join('trained_models', 'im2state')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    net = CNN(3, len(args.state_indices))
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None, args.add_timestep, 'cpu',
                        False, None, no_norm=True)
    env.close()
    low = env.observation_space.low[args.state_indices]
    high = env.observation_space.high[args.state_indices]

    p = np.random.permutation(len(images))
    x = images[p]
    y = normalise_coords(positions, low, high)[p]

    batch_size = 128
    num_test_examples = 512

    train_x = torch.Tensor(x[num_test_examples:])
    train_y = torch.Tensor(y[num_test_examples:])
    test_x = torch.Tensor(x[:num_test_examples])
    test_y = torch.Tensor(y[:num_test_examples])

    train_loss = []
    test_loss = []
    min_test_loss = math.inf

    updates_with_no_improvement = 0

    # run the main training loop
    epochs = 0
    while updates_with_no_improvement < 5:
        epochs += 1
        losses = []
        for batch_idx in range(0, len(train_x), batch_size):
            actual_y = net(train_x[batch_idx:batch_idx + batch_size])
            loss = criterion(actual_y, train_y[batch_idx:batch_idx + batch_size])
            losses += [loss.item()]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += [np.mean(losses)]
        test_loss += [criterion(net(test_x), test_y).item()]
        if test_loss[-1] < min_test_loss:
            updates_with_no_improvement = 0
            min_test_loss = test_loss[-1]
        else:
            updates_with_no_improvement += 1

        if epochs % args.save_interval == 0 or updates_with_no_improvement == 1:
            save_model = net
            if args.cuda:
                save_model = copy.deepcopy(net).cpu()

            if updates_with_no_improvement <= 1:
                torch.save(save_model, os.path.join(save_path, args.save_as + ".pt"))

        if epochs % args.log_interval == 0 or updates_with_no_improvement == 5:
            plt.figure()
            plt.plot(range(epochs), train_loss, label="Training Loss")
            plt.plot(range(epochs), test_loss,  label="Test Loss")
            plt.legend()
            plt.show()
            print(f"Training epoch {epochs} - validation loss: {test_loss[-1]}")


if __name__ == "__main__":
    main()
