import os
import platform

import numpy as np
import torch
from PIL import Image

from a2c_ppo_acktr.arguments import get_args
from envs.DRRewardEnvs import DRSparseEnv
from envs.DishRackEnv import rack_lower, rack_upper
from envs.envs import make_vec_envs

from tqdm import tqdm

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

save_root = '' if platform.system() == 'Darwin' else '/vol/bitbucket2/hu115/'


# Script for rolling out a trained policy to gather images used to train an end-to-end controller
# (see train_e2e.py)
def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    policies = torch.load(os.path.join(args.load_dir, 'ppo', args.initial_policy + ".pt"))

    envs = make_vec_envs(DRSparseEnv, 'dish_rack_vis', args.seed + 1000, args.num_processes,
                         args.gamma, args.log_dir, device, False, policies, no_norm=True,
                         show=(args.num_processes == 1))

    null_action = torch.zeros((args.num_processes, envs.action_space.shape[0]))
    save_path = os.path.join(save_root, f'training_data/{args.seed}')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    low = rack_lower
    high = rack_upper

    envs.get_images(mode='activate')
    image_paths = []
    actions = np.zeros((args.num_steps, 6))
    joint_targets = np.zeros((args.num_steps, 7))
    joint_angles = np.zeros((args.num_steps, 7))

    obs = envs.reset()

    for i in tqdm(range(args.num_steps // args.num_processes)):
        start_index = args.num_processes * i
        for j, image in enumerate(envs.get_images()):
            file_num = args.seed * 100000 + args.num_processes * i + j
            Image.fromarray(image, 'RGB').save(os.path.join(save_path, f'{file_num}.png'))
            image_paths += [f'{file_num}.png']
        joint_angles[start_index:start_index + args.num_processes] = obs[:, :7]

        obs = envs.step(null_action)[0]

        action = np.array(envs.get_images(mode="action"))
        actions[start_index:start_index + args.num_processes] = action
        target = np.array(envs.get_images(mode="joint_target_pos"))
        joint_targets[start_index:start_index + args.num_processes] = target

    envs.close()

    torch.save([image_paths, joint_angles, actions, low, high],
               os.path.join(save_path,f'{args.initial_policy}_{args.num_steps}_{args.seed}_e2e.pt'))


if __name__ == "__main__":
    main()
