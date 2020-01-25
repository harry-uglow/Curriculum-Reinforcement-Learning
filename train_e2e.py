import copy
import math
import os

import numpy as np
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from a2c_ppo_acktr.arguments import get_args
from e2e.dataset import E2EDataset
from e2e.model import E2ECNN

from pose_estimator.utils import normalise_target

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# From https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    """
    This is a pytorch generic function that takes a data.Dataset object and splits it to
    validation and training efficiently.
    """
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen // split_fold
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid


# Train an end-to-end (image + partial state -> action) controller to approximate a trained
# full-state policy. These end-to-end approximations were used on the real robot during the
# project.
def main():
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.device_num}" if args.cuda else "cpu")
    print(device)

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    target_transform = lambda y: normalise_target(y, np.array([-0.3] * 7), np.array([0.3] * 7))
    dataset = E2EDataset(os.path.join(args.load_dir, args.env_name + ".pt"), args.load_dir,
                         transform, target_transform)
    print("Loaded")

    train, valid = train_valid_split(dataset, random_seed=1053831)

    save_path = os.path.join('trained_models', 'pe')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    batch_size = 400
    num_train_examples = len(train)
    num_test_examples = len(valid)
    print(len(dataset))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=2)

    net = E2ECNN(3, 7)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_loss_x_axis = []
    train_loss = []
    test_loss = []
    min_test_loss = math.inf

    updates_with_no_improvement = 0

    # run the main training loop
    epochs = 0
    while updates_with_no_improvement < 5:
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            image = batch['image'].to(device)
            angles = batch['angles'].to(device)
            action = batch['action'].to(device)

            output = net((image, angles))
            loss = criterion(output, action)

            train_loss += [loss.item()]
            train_loss_x_axis += [epochs + (batch_idx * batch_size + batch_size) / num_train_examples]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs += 1

        loss = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(valid_loader)):
                image = batch['image'].to(device)
                angles = batch['angles'].to(device)
                action = batch['action'].to(device)

                test_output = net((image, angles))
                loss += criterion(test_output, action).item()

            test_loss += [loss / (num_test_examples // batch_size)]
        if test_loss[-1] < min_test_loss:
            updates_with_no_improvement = 0
            min_test_loss = test_loss[-1]

            save_model = net
            if args.cuda:
                save_model = copy.deepcopy(net).cpu()
            torch.save(save_model, os.path.join(save_path, args.save_as + ".pt"))
            print("Saved")
        else:
            updates_with_no_improvement += 1

        if epochs % args.log_interval == 0 or updates_with_no_improvement == 5:
            fig = plt.figure()
            plt.plot(train_loss_x_axis, train_loss, label="Training Loss")
            plt.plot(range(1, epochs + 1), test_loss,  label="Test Loss")
            plt.legend()
            plt.savefig(f'imgs/{args.save_as}.png')
            plt.close(fig)
            print(f"Training epoch {epochs} - validation loss: {test_loss[-1]}")

    print("Finished training")


if __name__ == "__main__":
    main()
