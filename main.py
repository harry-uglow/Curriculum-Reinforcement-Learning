import copy
import glob
import math
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
from envs.pipelines import pipelines

args = get_args()
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

use_metric = args.trg_succ_rate is not None
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(env, scene_path):
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    save_path = os.path.join(args.save_dir, args.algo)

    eval_log_dir = args.log_dir + "_eval"
    eval_x = []
    eval_y = []

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

    envs = make_vec_envs(env, scene_path, args.seed, args.num_processes, args.gamma, args.log_dir,
                         device, False, initial_policies, pose_estimator=pose_estimator,
                         init_control=not args.dense_ip)
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

    episode_rewards = deque(maxlen=64)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    total_num_steps = 0
    j = 0
    max_succ = 0
    max_mean_rew = - math.inf
    evals_without_improv = 0

    start = time.time()
    start_update = start
    while (not use_metric and j < num_updates) or (use_metric and max_succ < args.trg_succ_rate):
        if args.eval_interval is not None and j % args.eval_interval == 0:
            print("Evaluating current policy...")
            i = 0
            total_successes = 0
            max_trials = 50
            eval_episode_rewards = []
            eval_recurrent_hidden_states = torch.zeros(
                args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)
            while i + args.num_processes <= max_trials:

                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                obs, _, dones, infos = envs.step(action)

                if np.all(dones):  # Rigid - assumes episodes are fixed length
                    rews = []
                    for info in infos:
                        rews.append(info['rew_success'])
                        eval_episode_rewards.append(info['episode']['r'])
                    i += args.num_processes
                    rew = sum([int(rew > 0) for rew in rews])
                    total_successes += rew

            p_succ = (100 * total_successes / i)
            eval_x += [total_num_steps]
            eval_y += [p_succ]

            end = time.time()
            print(f"Evaluation: {total_successes} successful out of {i} episodes - "
                  f"{p_succ:.2f}% successful. Eval length: {end - start_update}")
            torch.save([eval_x, eval_y], os.path.join(args.save_as + "_eval.pt"))
            start_update = end

            mean_ep_rew = np.mean(eval_episode_rewards)
            if p_succ > max_succ:
                max_succ = p_succ
                max_mean_rew = mean_ep_rew
                evals_without_improv = 0
            elif p_succ == max_succ and mean_ep_rew > max_mean_rew:
                print("Same success rate, higher reward")
                max_mean_rew = mean_ep_rew
                evals_without_improv = 0
            else:
                evals_without_improv += 1

            if evals_without_improv == 5 or max_succ >= args.trg_succ_rate:
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None),
                              initial_policies]
                extra = "_final" if evals_without_improv == 5 else ""
                torch.save(save_model, os.path.join(save_path, args.save_as + f"{extra}.pt"))
                break

        # save for every interval-th episode or for the last epoch
        if ((not use_metric and (j % args.save_interval == 0 or j == num_updates - 1))
                or (use_metric and evals_without_improv == 0)) and args.save_dir != "":
            os.makedirs(save_path, exist_ok=True)

            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            if pose_estimator is not None:
                save_model = [save_model, pose_estimator, initial_policies]
            else:
                save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None),
                              initial_policies]

            torch.save(save_model, os.path.join(save_path, args.save_as + ".pt"))
            # torch.save(save_model, os.path.join(save_path, args.save_as + f"{j * args.num_processes * args.num_steps}.pt"))

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

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
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

        if args.vis and (j % args.vis_interval == 0 or (not use_metric and j == num_updates - 1)):
            try:
                # Sometimes monitor doesn't properly flush the outputs
                visdom_plot(args.log_dir, args.save_as, args.algo, total_num_steps)
            except IOError:
                pass

        j += 1

    if use_metric:
        if max_succ >= args.trg_succ_rate:
            print(f"Achieved greater than {args.trg_succ_rate}% success, advancing curriculum.")
        else:
            print(f"Policy converged with max success rate < {args.trg_succ_rate}%")
    # Copy logs to permanent location so new graphs can be drawn.
    copy_tree(args.log_dir, os.path.join('logs', args.save_as))
    envs.close()
    return total_num_steps


def train_with_metric(pipeline, train):
    if args.use_linear_clip_decay:
        raise ValueError("Cannot use clip decay with unbounded metric-based training length.")
    if args.eval_interval is None:
        raise ValueError("Need to set eval_interval to evaluate success rate")
    args.use_linear_lr_decay = False

    training_lengths = []
    save_path = os.path.join(args.save_dir, args.algo)
    base = args.save_as
    for i in range(0, args.num_seeds * 16, 16):
        args.seed = i
        print(f"Using seed {args.seed}")
        training_lengths += [train(pipeline, f"{base}_{i}")]
        torch.save(training_lengths,
                   os.path.join(save_path, f"{base}_{args.pipeline}_train_lengths.pt"))

    print(training_lengths)
    total_train_times = [sum(lengths) for lengths in training_lengths]
    print(total_train_times)
    torch.save([total_train_times, training_lengths],
               os.path.join(save_path, f"{base}_{args.pipeline}_train_lengths.pt"))
    return total_train_times, training_lengths


def execute_curriculum(pipeline, save_base):
    training_lengths = []
    criteria_string = f"until {args.trg_succ_rate}% successful" if use_metric \
        else f"for {args.num_env_steps} timesteps"
    for scene in pipeline['curriculum']:
        print(f"Training {scene} {criteria_string}")
        args.save_as = f'{save_base}_{scene}'
        training_lengths += [main(pipeline['sparse'], scene)]
        args.reuse_residual = True
        args.initial_policy = args.save_as
    scene = pipeline['task']
    print(f"Training on {scene} full task")
    args.save_as = f'{save_base}_{scene}'
    args.trg_succ_rate = 101  # Does not affect fixed length curriculum
    training_lengths += [main(pipeline['sparse'], scene)]
    return training_lengths


def train_baseline(pipeline, save_base):
    training_lengths = []
    scene = pipeline['task']
    print(f"Training {scene} until {args.trg_succ_rate}% successful with dense rewards")
    args.save_as = f'{save_base}_dense_{scene}'
    training_lengths += [main(pipeline['dense'], scene)]
    args.initial_policy = args.save_as
    print(f"Training on {scene} until convergence")
    args.save_as = f'{save_base}_sparse_{scene}'
    args.trg_succ_rate = 101
    training_lengths += [main(pipeline['sparse'], scene)]
    return training_lengths


if __name__ == "__main__":
    if args.scene_name is not None:
        main(None, args.scene_name)  # TODO
    elif args.dense_ip:
        train_with_metric(pipelines[args.pipeline], train_baseline)
    elif use_metric:
        train_with_metric(pipelines[args.pipeline], execute_curriculum)
    else:
        execute_curriculum(pipelines[args.pipeline], args.save_as)
