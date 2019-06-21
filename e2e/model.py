import torch
import torch.nn as nn
from torchvision import transforms


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class E2ECNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(E2ECNN, self).__init__()
        self._output_size = num_outputs

        self.conv_layers = nn.Sequential(  # 120 x 120
            (nn.Conv2d(num_inputs, 64, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(64, 64, 3, padding=1, stride=2)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 60
            (nn.Conv2d(64, 128, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(128, 128, 3, padding=1, stride=2)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 30
            (nn.Conv2d(128, 256, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(256, 256, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(256, 256, 3, padding=1, stride=2)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 15
            (nn.Conv2d(256, 512, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(512, 512, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(512, 512, 3, padding=1, stride=2)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 8
            (nn.Conv2d(512, 512, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(512, 512, 3, padding=1)),
            nn.ReLU(inplace=True),
            (nn.Conv2d(512, 512, 3, padding=1, stride=2)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 4
        )

        self.flatten = Flatten()

        self.fc_layers = nn.Sequential(
            (nn.Linear(4 * 4 * 512 + 7, 256)),
            nn.ReLU(inplace=True),
            (nn.Linear(256, 64)),
            nn.ReLU(inplace=True),
            (nn.Linear(64, num_outputs))
        )

        self.train()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x):
        images, angles = x
        conv_output = self.flatten(self.conv_layers(images))

        x = torch.cat((conv_output, angles), dim=1)
        x = self.fc_layers(x)
        return x

    def predict(self, images, angles):
        x = torch.Tensor(images.cpu())
        for i in range(x.size(0)):
            x[i] = self.normalize(x[i] / 255.0)
        return self.forward((x.to(images.device), angles))
