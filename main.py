import copy
import glob
import os
import time
from collections import deque
from distutils.dir_util import copy_tree

import numpy as np
import torch

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot

args = get_args()
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(scene_path):
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    eval_log_dir = args.log_dir + "_eval"

    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    initial_policies = torch.load(os.path.join(args.load_dir, args.algo,
                                               args.initial_policy + ".pt")) \
        if args.initial_policy else None

    if args.reuse_residual:
        residual, ob_rms, initial_policies = initial_policies
    else:
        residual = None
        ob_rms = None

    pose_estimator = torch.load(os.path.join(args.load_dir, "im2state",
                                             args.pose_estimator + ".pt")) \
        if args.pose_estimator else None

    envs = make_vec_envs(scene_path, args.seed, args.num_processes, args.gamma, args.log_dir,
                         args.add_timestep, device, False, initial_policies,
                         pose_estimator=pose_estimator, e2e=args.e2e)
    if args.reuse_residual:
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

    base_kwargs = {'recurrent': args.recurrent_policy}
    base = residual.base if args.reuse_residual else None
    dist = residual.dist if args.reuse_residual else None
    actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs=base_kwargs,
                          zero_last_layer=initial_policies is not None, base=base, dist=dist)
    actor_critic.to(device)

    print(args.e2e)
    print(not args.reuse_residual and args.e2e)
    if not args.reuse_residual and args.e2e:
        pretrained_cnn = torch.load(os.path.join('trained_models/im2state',
                                                 "full_vgg16_16_diag_ren_l1_rpt.pt"))
        actor_critic.base.conv_layers.load_state_dict(pretrained_cnn.conv_layers.state_dict())
        #actor_critic.base.conv_layers.eval()

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                         max_grad_norm=args.max_grad_norm,
                         burn_in=initial_policies is not None and not args.reuse_residual)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.num_processes * args.num_steps // 48)  # ep_len = 48
    max_min_rew = 0
    max_median_rew = 0

    start = time.time()
    start_update = start
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        median_rew = np.median(episode_rewards)
        min_rew = np.min(episode_rewards)
        max_median_rew = max(max_median_rew, median_rew)
        max_min_rew = max(max_min_rew, min_rew)

        # save for every interval-th episode or for the last epoch
        if min_rew == max_min_rew and args.save_dir != "":
            print("Saving")
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # TODO: Save better
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            if pose_estimator is not None:
                save_model = [save_model, pose_estimator, initial_policies]
            else:
                save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None),
                              initial_policies]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            print("Update length: ", end - start_update)
            start_update = end

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, 1, args.gamma, eval_log_dir,
                args.add_timestep, device, True, show=True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 1:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                           for done_ in done],
                                           dtype=torch.float32,
                                           device=device)

                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if args.vis and (j % args.vis_interval == 0 or j == num_updates - 1):
            try:
                # Sometimes monitor doesn't properly flush the outputs
                visdom_plot(args.log_dir, args.env_name, args.algo, args.num_env_steps)
            except IOError:
                pass
    # Copy logs to permanent location so new graphs can be drawn.
    copy_tree(args.log_dir, os.path.join('logs', args.env_name))
    envs.close()

def full_pipeline():
    main('reach_no_wall')
    args.initial_policy = args.env_name
    base_name = 'row'
    scene_names = [
        'reach_no_wall',
        'row_13',
        'row_19',
        'row_25',
        'row_31',
        'row_37',
        'reach_over_wall',
    ]
    args.num_steps = 200000
    for scene in scene_names:
        print(f"Training {scene} for {args.num_env_steps} timesteps")
        args.env_name = f'{base_name}_{scene}'
        if scene == 'reach_over_wall':
            args.num_steps = 4000000
        main(scene)
        args.reuse_residual = True
        args.initial_policy = args.env_name

scene_names = [
    'dish_rack',
]

if __name__ == "__main__":
    if args.reuse_residual:
        base_name = args.env_name
        base_ip = args.initial_policy
        for scene in scene_names:
            print(f"Training {scene} for {args.num_env_steps} timesteps")
            args.env_name = f'{base_name}_{scene}'
            main(scene)
            args.initial_policy = args.env_name
    else:
        main('dish_rack')
