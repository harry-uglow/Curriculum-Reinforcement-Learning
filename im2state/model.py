import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CNN, self).__init__()
        self._output_size = num_outputs

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(  # 64 x 64
            init_(nn.Conv2d(num_inputs, 32, 7, stride=3)),  # 20 x 20
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),  # 9 x 9
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),  # 7 x 7
            nn.ReLU(),
            init_(nn.Conv2d(32, 16, 3, stride=1)),  # 5 x 5
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(16 * 5 * 5, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, num_outputs))
        )

        self.train()

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x):
        # Normalise inputs
        x = self.main(x / 255.0)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
