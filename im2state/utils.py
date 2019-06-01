import numpy as np
import torch


def format_images(imgs):
    array = np.asarray(imgs)
    return array.transpose(0, 3, 1, 2)


# Normalise coordinates so all are in range [-1, 1]. Inputs must be ndarray.
def normalise_coords(coords, low, high):
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


def ren_loss(pred, actual):
    l1_translation_loss = torch.nn.L1Loss()(pred[:, :2], actual[:, :2])
    cosine_difference = torch.cos(pred[:, 2] - actual[:, 2])
    orientation_loss = torch.nn.L1Loss()(cosine_difference, torch.ones_like(cosine_difference))
    return torch.add(l1_translation_loss, orientation_loss)
