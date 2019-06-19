import os

import torch
from torch.utils.data import Dataset
from skimage import io


class E2EDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file, image_dir):
        """
        Args:
            data_file (string): Path to the saved image names and associated data.
            image_dir (string): Directory with all the images.
        """
        image_names, angles, actions = torch.load(data_file)
        self.image_names = image_names
        self.angles = angles
        self.actions = actions
        self.root_dir = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = io.imread(img_name)
        angles = self.angles[idx]
        action = self.actions[idx]
        sample = {'image': image, 'angles': angles, 'action': action}

        return sample
