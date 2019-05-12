import numpy as np


def format_images(imgs):
    array = np.asarray(imgs)
    return array.transpose(0, 3, 1, 2)
