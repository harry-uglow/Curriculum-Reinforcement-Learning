import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class InitialPolicy(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size=50):
        super(InitialPolicy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Data should be normalised
def train_nn(net, train_x, train_y, lr=0.1, num_epochs=10):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)

    # run the main training loop
    epochs = 0
    while epochs < num_epochs:
        epochs += 1
        actual_y = net(train_x)
        loss = criterion(actual_y, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net
