import os
import platform

import numpy as np
import torch
from PIL import Image

from a2c_ppo_acktr.arguments import get_args
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


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    policies = torch.load(os.path.join(args.load_dir, 'ppo', args.env_name + ".pt"))

    envs = make_vec_envs('dish_rack', args.seed + 1000, args.num_processes, args.gamma,
                         args.log_dir, args.add_timestep, device, False, policies,
                         no_norm=True, show=(args.num_processes == 1))

    null_action = torch.zeros((args.num_processes, envs.action_space.shape[0]))
    save_path = os.path.join(save_root, 'training_data')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    low = rack_lower
    high = rack_upper

    envs.get_images(mode='activate')
    images = []
    rel_positions = np.zeros((args.num_steps, len(args.state_indices)))
    abs_positions = np.zeros((args.num_steps, len(args.state_indices) - 1))
    actions = np.zeros((args.num_steps, 7))
    joint_angles = np.zeros((args.num_steps, 7))

    obs = envs.reset()

    for i in tqdm(range(args.num_steps // args.num_processes)):
        start_index = args.num_processes * i
        rel_obs = obs[:, args.state_indices]
        abs_obs = np.array(envs.get_images(mode="target"))
        images += [Image.fromarray(img, 'RGB') for img in envs.get_images()]
        rel_positions[start_index:start_index + args.num_processes] = rel_obs
        abs_positions[start_index:start_index + args.num_processes] = abs_obs
        joint_angles[start_index:start_index + args.num_processes] = obs[:, :7]

        obs = envs.step(null_action)[0]

        action = np.array(envs.get_images(mode="action"))
        actions[start_index:start_index + args.num_processes] = action

        if i % 1000 == 999:
            j = start_index + args.num_processes
            torch.save([images[:j], abs_positions[:j], rel_positions[:j], low, high,
                        joint_angles[:j], actions[:j]],
                       os.path.join(save_path, f'{args.env_name}_{args.num_steps}_e2e.pt'))

    envs.close()

    torch.save([images, abs_positions, rel_positions, low, high],
               os.path.join(save_path, f'{args.env_name}_{args.num_steps}_e2e.pt'))


if __name__ == "__main__":
    main()
