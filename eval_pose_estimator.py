import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--model-name', default='rel_new_angle')
parser.add_argument('--dataset', default='rel_dish_rack_nr_128_8192')
parser.add_argument('--num-examples', type=int, default=256)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()


def eval_pose_estimator(load_path, x, y):
    print("Evaluating")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    net = torch.load(load_path)
    net.to(device)
    x.to(device)
    y = y.to("cpu").numpy()
    net.eval()
    with torch.no_grad():
        distances = []
        thetas = []
        for i in tqdm(range(x.size(0))):
            actual_y = net(x[i].unsqueeze(0)).squeeze().cpu().numpy()
            pred_y = y[i]
            distances += [np.linalg.norm(pred_y[:3] - actual_y[:3])]
            thetas += [np.abs(pred_y[3] - actual_y[3])]
        print(f"Mean distance error: {(1000 * sum(distances) / len(distances))}mm")
        print(f"Mean rotational error: {(sum(thetas) / len(thetas))} radians")


if __name__ == '__main__':
    images, positions, state_to_estimate, low, high = torch.load(
        os.path.join('./training_data', args.dataset + ".pt"))
    images = np.transpose([np.array(img) for img in images], (0, 3, 1, 2))
    model_dir_path = os.path.join('trained_models', 'im2state')
    load_path = os.path.join(model_dir_path, args.model_name + ".pt")

    p = np.random.permutation(len(images))

    x = torch.Tensor(images[p][:args.num_examples])
    y = torch.Tensor(positions[p][:args.num_examples])
    eval_pose_estimator(load_path, x, y)
