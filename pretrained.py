import os

import torch
import torchvision.models as models

from im2state.model import PoseEstimator

vgg16 = models.vgg16(pretrained=True)
relevant_vgg_layers = [feature for feature in vgg16.features]

net = PoseEstimator(3, 3, [7, 8, 9])
relevant_PE_layers = [layer for layer in net.main]

for i in range(24):
    relevant_PE_layers[i].load_state_dict(relevant_vgg_layers[i].state_dict())
torch.save(net.state_dict(), os.path.join('', "vgg16.pt"))