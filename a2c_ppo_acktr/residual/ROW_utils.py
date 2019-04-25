import math

import numpy as np

from a2c_ppo_acktr.envs.SawyerEnv import normalise_angles


def train_initial_policy(env):
    x, y = env.get_initial_data()
    num_inputs = len(x[0])
    num_outputs = len(y[0])

    p = env.np_random.permutation(len(x))
    env.initial_policy.train_net(x[p], y[p])

    null_action = np.array([0.] * num_outputs)

    min_loss = math.inf
    episodes = 0
    episodes_with_no_improvement = 0

    # Use DAgger until stabilised
    while episodes < 10 or episodes_with_no_improvement < 2:
        env.reset()
        done = False
        x_to_append = np.zeros((env.ep_len, num_inputs))
        y_to_append = np.zeros((env.ep_len, num_outputs))
        while not done:
            _, _, done, _ = env.step(null_action)

            new_x, new_y = env.solve_ik()
            if len(new_y) != 0:
                x_to_append[env.timestep - 1] = new_x[0]
                y_to_append[env.timestep - 1] = new_y[0]

        x = np.append(x, normalise_angles(x_to_append), axis=0)
        y = np.append(y, y_to_append, axis=0)
        p = env.np_random.permutation(len(x))
        loss = env.initial_policy.train_net(x[p], y[p])
        print(f"Training initial policy - DAgger episode {episodes} - loss: {loss}")

        if loss < min_loss:
            episodes_with_no_improvement = 0
            min_loss = loss
        elif episodes >= 10:
            episodes_with_no_improvement += 1
        episodes += 1

    return env.initial_policy
