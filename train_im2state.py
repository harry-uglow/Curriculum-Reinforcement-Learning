import copy
import math
import os

import numpy as np
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from a2c_ppo_acktr.arguments import get_args
from eval_pose_estimator import eval_pose_estimator
from im2state.model import PoseEstimator

from im2state.utils import normalise_coords, unnormalise_y, custom_loss

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.device_num}" if args.cuda else "cpu")

    images, abs_positions, rel_positions, low, high = torch.load(
        os.path.join(args.load_dir, args.env_name + ".pt"))
    print("Loaded")
    low = torch.Tensor(low).to(device)
    high = torch.Tensor(high).to(device)
    images = np.transpose([np.array(img) for img in images], (0, 3, 1, 2))

    save_path = os.path.join('trained_models', 'im2state')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    positions = rel_positions if args.rel else abs_positions
    pretrained_name = 'vgg16_4out.pt' if args.rel else 'vgg16_3out.pt'

    net = PoseEstimator(3, positions.shape[1])
    net.load_state_dict(torch.load(os.path.join('trained_models/pretrained/', pretrained_name)))
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = custom_loss

    np_random = np.random.RandomState()
    np_random.seed(1053831)
    p = np_random.permutation(len(images))
    x = images[p]
    y = positions[p]

    num_test_examples = 256
    batch_size = 50

    train_x = torch.Tensor(x[num_test_examples:])
    train_y = torch.Tensor(y[num_test_examples:]).to(device)
    test_x = torch.Tensor(x[:num_test_examples])
    test_y = torch.Tensor(y[:num_test_examples]).to(device)

    num_train_examples = train_x.size(0)
    print("Normalising")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_train_examples):
        train_x[i] = normalize(train_x[i] / 255.0)
    for i in range(num_test_examples):
        test_x[i] = normalize(test_x[i] / 255.0)
    train_x = train_x.to(device)
    test_x = test_x.to(device)

    train_loss_x_axis = []
    train_loss = []
    test_loss = []
    min_test_loss = math.inf

    updates_with_no_improvement = 0

    # run the main training loop
    epochs = 0
    while updates_with_no_improvement < 5:
        for batch_idx in tqdm(range(0, num_train_examples, batch_size)):
            output = net(train_x[batch_idx:batch_idx + batch_size])
            pred_y = output if args.rel else unnormalise_y(output, low, high)
            loss = criterion(pred_y, train_y[batch_idx:batch_idx + batch_size])
            train_loss += [loss.item()]
            train_loss_x_axis += [epochs + (batch_idx + batch_size) / num_train_examples]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs += 1

        with torch.no_grad():
            test_output = net(test_x)
            test_output = test_output if args.rel else unnormalise_y(test_output, low, high)
            test_loss += [criterion(test_output, test_y).item()]
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
    eval_pose_estimator(os.path.join(save_path, args.save_as + ".pt"), device, test_x, test_y,
                        low if not args.rel else None, high if not args.rel else None)


if __name__ == "__main__":
    main()
