import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PoseEstimator(nn.Module):
    def __init__(self, num_inputs, num_outputs, state_to_estimate):
        super(PoseEstimator, self).__init__()
        self._output_size = num_outputs
        self.state_to_estimate = state_to_estimate

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(  # 128 x 128
            init_(nn.Conv2d(num_inputs, 64, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64
            init_(nn.Conv2d(64, 128, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(128, 128, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32
            init_(nn.Conv2d(128, 256, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(256, 256, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(256, 256, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16
            init_(nn.Conv2d(256, 512, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(512, 512, 3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(512, 512, 3)),  # 14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7
            Flatten(),
            init_(nn.Linear(7 * 7 * 512, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, 64)),
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
