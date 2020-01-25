import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from pose_estimator.utils import unnormalise_y

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--model-name', default='rel_new_angle')
parser.add_argument('--dataset', default='rel_dish_rack_nr_128_8192')
parser.add_argument('--num-examples', type=int, default=256)


def eval_pose_estimator(load_path, device, x, y, low, high):
    print("Evaluating")
    net = torch.load(load_path)
    net.to(device)
    x.to(device)
    y = y.cpu().numpy()
    net.eval()
    with torch.no_grad():
        distances = []
        thetas = []
        for x_, y_ in tqdm(zip(x, y)):
            output = net.predict(x_.unsqueeze(0))
            normed = output if low is None else unnormalise_y(output, low, high)
            pred_y = normed.squeeze().cpu().numpy()
            actual_y = y_
            distances += [np.linalg.norm(pred_y[:-1] - actual_y[:-1])]
            thetas += [np.abs(pred_y[-1] - actual_y[-1])]
        print(f"Mean distance error: {(1000 * sum(distances) / len(distances))} mm")
        print(f"Mean rotational error: {(sum(thetas) / len(thetas))} radians")


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    images, positions, state_to_estimate, low, high = torch.load(
        os.path.join('./training_data', args.dataset + ".pt"))
    images = np.transpose([np.array(img) for img in images], (0, 3, 1, 2))
    model_dir_path = os.path.join('trained_models', 'pe')
    load_path = os.path.join(model_dir_path, args.model_name + ".pt")

    p = np.random.permutation(len(images))

    x = torch.Tensor(images[p][:args.num_examples])
    y = torch.Tensor(positions[p][:args.num_examples])
    eval_pose_estimator(load_path, device, x, y, torch.Tensor(low), torch.Tensor(high))
