import numpy as np
from torch import nn, cos, ones_like


def format_images(imgs):
    array = np.asarray(imgs)
    return array.transpose(0, 3, 1, 2)


# Normalise target value so all are in range [-1, 1]. Inputs must be ndarray.
def normalise_target(coords, low, high):
    nlow = len(low)
    assert nlow == len(high)
    assert nlow == coords.shape[-1] or nlow == 1
    return (((coords - low) / (high - low)) * 2) - 1


# Transform y in [-1, 1] to y in [low, high]. All inputs must be ndarray.
def unnormalise_y(y, low, high):
    nlow = len(low)
    assert nlow == len(high)
    assert nlow == y.shape[-1] or nlow == 1
    return (((y + 1) / 2) * (high - low)) + low


def custom_loss(pred, actual):
    translation_loss = nn.L1Loss()(pred[:, :-1], actual[:, :-1])
    cos_diff = cos(pred[:, -1] - actual[:, -1])
    orientation_loss = nn.L1Loss()(cos_diff, ones_like(cos_diff).to(cos_diff.device))
    return translation_loss + orientation_loss
