import copy
import math
import os

import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from a2c_ppo_acktr.arguments import get_args
from eval_pose_estimator import eval_pose_estimator
from pose_estimator.model import PoseEstimator
from pose_estimator.utils import unnormalise_y, custom_loss

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Used to train pose estimators (image -> state) so that a (state -> action) could be used in a
# real environment. Not used recently as found to be less effective than train_e2e.py
def main():
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{args.device_num}" if args.cuda else "cpu")

    images, abs_positions, rel_positions, low, high= torch.load(
        os.path.join(args.load_dir, args.env_name + ".pt"))
    print("Loaded")
    low = torch.Tensor(low).to(device)
    high = torch.Tensor(high).to(device)

    save_path = os.path.join('trained_models', 'pe')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    positions = rel_positions if args.rel else abs_positions
    print(positions.shape[1])

    net = PoseEstimator(3, positions.shape[1])
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = custom_loss

    num_samples = len(images)

    np_random = np.random.RandomState()
    np_random.seed(1053831)
    p = np_random.permutation(num_samples)
    images = np.transpose([np.array(img) for img in images], (0, 3, 1, 2))
    positions = positions[p]

    num_test_examples = num_samples // 10
    num_train_examples = num_samples - num_test_examples
    batch_size = 100

    test_indices = p[:num_test_examples]
    train_indices = p[num_test_examples:]
    test_x = images[test_indices]

    test_y = positions[:num_test_examples]
    train_y = positions[num_test_examples:]

    train_loss_x_axis = []
    train_loss = []
    test_loss = []
    min_test_loss = math.inf

    updates_with_no_improvement = 0

    # run the main training loop
    epochs = 0
    while updates_with_no_improvement < 5:
        for batch_idx in tqdm(range(0, num_train_examples, batch_size)):
            indices = train_indices[batch_idx:batch_idx + batch_size]
            train_x = images[indices]

            output = net.predict(torch.Tensor(train_x).to(device))
            pred_y = output if args.rel else unnormalise_y(output, low, high)
            loss = criterion(pred_y,
                             torch.Tensor(train_y[batch_idx:batch_idx + batch_size]).to(device))
            train_loss += [loss.item()]
            train_loss_x_axis += [epochs + (batch_idx + batch_size) / num_train_examples]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs += 1

        loss = 0
        with torch.no_grad():
            for batch_idx in tqdm(range(0, num_test_examples, batch_size)):
                test_output = net.predict(torch.Tensor(test_x[batch_idx:batch_idx + batch_size]).to(device))
                test_output = test_output if args.rel else unnormalise_y(test_output, low, high)
                loss += criterion(test_output,
                             torch.Tensor(test_y[batch_idx:batch_idx + batch_size]).to(device)).item()

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
    eval_pose_estimator(os.path.join(save_path, args.save_as + ".pt"), device, torch.Tensor(test_x).to(device),
            torch.Tensor(test_y).to(device),
                        low if not args.rel else None, high if not args.rel else None)


if __name__ == "__main__":
    main()
