from __future__ import absolute_import
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description=u'RL')
    parser.add_argument(u'--algo', default=u'ppo',
                        help=u'algorithm to use: a2c | ppo | acktr')
    parser.add_argument(u'--lr', type=float, default=1e-4,
                        help=u'learning rate (default: 7e-4)')
    parser.add_argument(u'--eps', type=float, default=1e-5,
                        help=u'RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(u'--alpha', type=float, default=0.99,
                        help=u'RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(u'--gamma', type=float, default=0.99,
                        help=u'discount factor for rewards (default: 0.99)')
    parser.add_argument(u'--use-gae', action=u'store_true', default=True,
                        help=u'use generalized advantage estimation')
    parser.add_argument(u'--tau', type=float, default=0.95,
                        help=u'gae parameter (default: 0.95)')
    parser.add_argument(u'--entropy-coef', type=float, default=0,
                        help=u'entropy term coefficient (default: 0.01)')
    parser.add_argument(u'--value-loss-coef', type=float, default=0.5,
                        help=u'value loss coefficient (default: 0.5)')
    parser.add_argument(u'--max-grad-norm', type=float, default=0.5,
                        help=u'max norm of gradients (default: 0.5)')
    parser.add_argument(u'--seed', type=int, default=1,
                        help=u'random seed (default: 1)')
    parser.add_argument(u'--cuda-deterministic', action=u'store_true', default=False,
                        help=u"sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(u'--num-processes', type=int, default=16,
                        help=u'how many training CPU processes to use (default: 16)')
    parser.add_argument(u'--num-steps', type=int, default=5,
                        help=u'number of forward steps in A2C (default: 5)')
    parser.add_argument(u'--ppo-epoch', type=int, default=10,
                        help=u'number of ppo epochs (default: 4)')
    parser.add_argument(u'--num-mini-batch', type=int, default=32,
                        help=u'number of batches for ppo (default: 32)')
    parser.add_argument(u'--clip-param', type=float, default=0.2,
                        help=u'ppo clip parameter (default: 0.2)')
    parser.add_argument(u'--log-interval', type=int, default=1,
                        help=u'log interval, one log per n updates (default: 10)')
    parser.add_argument(u'--save-interval', type=int, default=20,
                        help=u'save interval, one save per n updates (default: 100)')
    parser.add_argument(u'--eval-interval', type=int, default=None,
                        help=u'eval interval, one eval per n updates (default: None)')
    parser.add_argument(u'--vis-interval', type=int, default=20,
                        help=u'vis interval, one log per n updates (default: 100)')
    parser.add_argument(u'--num-env-steps', type=int, default=10e6,
                        help=u'number of environment steps to train (default: 10e6)')
    parser.add_argument(u'--env-name', default=u'PongNoFrameskip-v4',
                        help=u'environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(u'--initial-policy', default=None,
                        help=u'initial policy to use, located in trained_models/ppo/{name}.pt')
    parser.add_argument(u'--pose-estimator', default=None,
                        help=u'pose estimator to use, located in trained_models/im2state/{name}.pt')
    parser.add_argument(u'--log-dir', default=u'/tmp/gym/',
                        help=u'directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(u'--save-dir', default=u'./trained_models/',
                        help=u'directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(u'--load-dir', default=u'./trained_models/',
                        help=u'directory to load initial policy from (default: ./trained_models/)')
    parser.add_argument(u'--no-cuda', action=u'store_true', default=False,
                        help=u'disables CUDA training')
    parser.add_argument(u'--device-num', type=int, default=0,
                        help=u'select CUDA device')
    parser.add_argument(u'--add-timestep', action=u'store_true', default=False,
                        help=u'add timestep to observations')
    parser.add_argument(u'--recurrent-policy', action=u'store_true', default=False,
                        help=u'use a recurrent policy')
    parser.add_argument(u'--use-linear-lr-decay', action=u'store_true', default=True,
                        help=u'use a linear schedule on the learning rate')
    parser.add_argument(u'--use-linear-clip-decay', action=u'store_true', default=False,
                        help=u'use a linear schedule on the ppo clipping parameter')
    parser.add_argument(u'--vis', action=u'store_true', default=False,
                        help=u'enable visdom visualization')
    parser.add_argument(u'--port', type=int, default=8097,
                        help=u'port to run the server on (default: 8097)')
    parser.add_argument(u'--e2e', action=u'store_true', default=False,
                        help=u'Train an e2e policy (do not use full state observations)')
    # TODO: Remove
    parser.add_argument(u'--save-as', default=u'i2s', help=u'Name to save im2state model under')
    parser.add_argument(u'--state-indices', nargs=u'+', type=int)
    parser.add_argument(u'--rel', action=u'store_true', default=False)
    parser.add_argument(u'--reuse-residual', action=u'store_true', default=False)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
