import numpy as np


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
