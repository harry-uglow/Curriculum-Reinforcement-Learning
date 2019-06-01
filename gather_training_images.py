import os
import platform

import numpy as np
import torch
from PIL import Image

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs.envs import make_vec_envs

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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir,
                         args.add_timestep, device, False, policies, no_norm=True)

    null_action = torch.zeros((args.num_processes, envs.action_space.shape[0]))

    envs.get_images(mode='activate')
    obs = envs.reset()
    images = [Image.fromarray(img, 'RGB') for img in envs.get_images()]
    positions = np.zeros((args.num_steps, len(args.state_indices)))
    positions[0: args.num_processes] = obs[:, args.state_indices]

    for i in tqdm(range(1, args.num_steps // args.num_processes)):

        obs, _, done, _ = envs.step(null_action)
        start_index = args.num_processes * i
        positions[start_index:start_index + args.num_processes] = obs[:, args.state_indices]

        images += [Image.fromarray(img, 'RGB') for img in envs.get_images()]

    envs.close()

    save_path = os.path.join(save_root, 'training_data')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    low = envs.observation_space.low[args.state_indices]
    high = envs.observation_space.high[args.state_indices]
    res = images[0].size[0]

    torch.save([images, positions, args.state_indices, low, high],
               os.path.join(save_path, f'{args.env_name}_{res}_{args.num_steps}_PIL.pt'))


if __name__ == "__main__":
    main()
