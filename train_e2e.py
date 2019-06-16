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
from e2e.model import E2ECNN
from envs.DRNoWaypointEnv import rack_lower
from envs.DishRackEnv import rack_upper
from eval_pose_estimator import eval_pose_estimator

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

    images, positions, _, _, _, joint_angles, actions = torch.load(
        os.path.join(args.load_dir, args.env_name + ".pt"))
    print("Loaded")
    low = np.array([-0.3] * 7)
    high = np.array([0.3] * 7)

    save_path = os.path.join('trained_models', 'im2state')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    net = E2ECNN(3, actions.shape[1] + positions.shape[1])
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    num_samples = len(images)
    print(num_samples)

    np_random = np.random.RandomState()
    np_random.seed(1053831)
    p = np_random.permutation(num_samples)
    images = np.transpose([np.array(img) for img in images], (0, 3, 1, 2))
    joint_angles = joint_angles[p]
    actions = normalise_coords(actions[p], low, high)
    positions = normalise_coords(positions[p], rack_lower, rack_upper)

    num_test_examples = num_samples // 10
    num_train_examples = num_samples - num_test_examples
    batch_size = 100

    print("Setting up data.")
    test_indices = p[:num_test_examples]
    train_indices = p[num_test_examples:]
    test_x = images[test_indices]

    y = np.append(actions, positions, axis=1)

    test_y = y[:num_test_examples]
    train_y = y[num_test_examples:]
    aux_test_x = joint_angles[:num_test_examples]
    aux_train_x = joint_angles[num_test_examples:]

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
            a = torch.Tensor(train_x).to(device)
            b = torch.Tensor(aux_train_x[batch_idx:batch_idx + batch_size]).to(device)
            pred_y = net.predict(a, b)
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
                test_output = net.predict(
                    torch.Tensor(test_x[batch_idx:batch_idx + batch_size]).to(device),
                    torch.Tensor(aux_test_x[batch_idx:batch_idx + batch_size]).to(device),
                )
                loss += criterion(
                    test_output,
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
    print("Evaluating")
    net = torch.load(os.path.join(save_path, args.save_as + ".pt"))
    net.to(device)
    x = torch.Tensor(test_x).to(device)
    aux_x = torch.Tensor(aux_test_x).to(device)
    y = unnormalise_y(test_y, low, high)
    net.eval()
    with torch.no_grad():
        distances = []
        thetas = []
        for x_, aux_x_, y_ in tqdm(zip(x, aux_x, y)):
            output = net.predict(x_.unsqueeze(0), aux_x_.unsqueeze(0))
            normed = output if low is None else unnormalise_y(output, low, high)
            pred_y = normed.squeeze().cpu().numpy()
            actual_y = y_
            distances += [np.linalg.norm(pred_y[:-1] - actual_y[:-1])]
            thetas += [np.abs(pred_y[-1] - actual_y[-1])]
        print(f"Mean distance error: {(1000 * sum(distances) / len(distances))} mm")
        print(f"Mean rotational error: {(sum(thetas) / len(thetas))} radians")


if __name__ == "__main__":
    main()
