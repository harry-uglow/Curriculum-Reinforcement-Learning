import math

import numpy as np
import torch

import torch.nn as nn
from torch import optim

from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.distributions import DiagGaussian


class InitialPolicy(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size=50):
        super(InitialPolicy, self).__init__()
        self.dist = DiagGaussian(hidden_size, num_outputs)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.actor(x)
        dist = self.dist(x)
        return dist.mode()

    # Data should be normalised
    def train_net(self, x, y, lr=0.05):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        num_test_examples = len(x) // 10

        train_x = torch.Tensor(x[num_test_examples:])
        train_y = torch.Tensor(y[num_test_examples:])
        test_x = torch.Tensor(x[:num_test_examples])
        test_y = torch.Tensor(y[:num_test_examples])

        min_test_loss = math.inf

        # run the main training loop
        epochs = 0
        while epochs < 100:
            epochs += 1
            actual_y = self(train_x)
            loss = criterion(actual_y, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_loss = criterion(self(test_x), test_y).item()
            if test_loss < min_test_loss:
                min_test_loss = test_loss

        return min_test_loss
