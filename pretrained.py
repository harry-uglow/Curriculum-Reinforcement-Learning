import os

import torch
import torchvision.models as models

from im2state.model import PoseEstimator

vgg16 = models.vgg16(pretrained=True)
relevant_vgg_layers = [feature for feature in vgg16.features]

num_outputs = 4
net = PoseEstimator(3, num_outputs)

net.conv_layers.load_state_dict(vgg16.features.state_dict())
torch.save(net.state_dict(), os.path.join('trained_models/pretrained',
                                          f"vgg16_{num_outputs}out.pt"))
