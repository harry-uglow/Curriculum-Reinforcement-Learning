import argparse

import torch


# TODO: Move arguments for gather_training_images / train_pose_estimator into their own parsers.
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='how many times to repeat the expirement with new seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=10,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--trg-succ-rate', type=int, default=None,
                        help='Require % success before stopping, instead of fixed training length.')
    parser.add_argument('--vis-interval', type=int, default=20,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--save-as', default=None,
                        help='base name for saved policies')
    parser.add_argument('--pipeline', default=None,
                        help='Pipeline of scenes to use CuRL on')
    parser.add_argument('--scene-name', default=None,
                        help='An individual scene to load, cannot be used with --pipeline')
    parser.add_argument('--first-stage', type=int, default=0,
                        help="Index of starting curriculum stage.")
    parser.add_argument('--initial-policy', default=None,
                        help='initial policy to use, located in trained_models/ppo/{name}.pt')
    parser.add_argument('--dense-ip', action='store_true', default=False,
                        help='use an IP trained with RL on dense rewards')
    parser.add_argument('--pose-estimator', default=None,
                        help='pose estimator to use, located in trained_models/pe/{name}.pt')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--load-dir', default='./trained_models/',
                        help='directory to load initial policy from (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device-num', type=int, default=0,
                        help='select CUDA device')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=True,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--e2e', action='store_true', default=False,
                        help='Train an e2e policy (do not use full state observations)')
    parser.add_argument('--reuse-residual', action='store_true', default=False)
    parser.add_argument('--state-indices', nargs='+', type=int)
    parser.add_argument('--rel', action='store_true', default=False)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
