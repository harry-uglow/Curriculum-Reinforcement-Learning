import math

import numpy as np

from a2c_ppo_acktr.envs.SawyerEnv import normalise_angles


def train_initial_policy(env):
    x, y = env.get_initial_data()

    p = env.np_random.permutation(len(x))
    env.initial_policy.train_net(x[p], y[p])

    null_action = np.array([0.] * len(y[0]))

    min_final_loss = math.inf
    episodes_with_no_improvement = 0

    # Use DAgger until stabilised
    while episodes_with_no_improvement < 2:
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(null_action)

            new_x, new_y = env.solve_ik()
            if len(new_x) != 0:
                x = np.append(x, normalise_angles(new_x[:-1]), axis=0)
                y = np.append(y, new_y, axis=0)

        p = env.np_random.permutation(len(x))
        final_loss = env.initial_policy.train_net(x[p], y[p])
        print(final_loss)

        if final_loss < min_final_loss:
            episodes_with_no_improvement = 0
        else:
            episodes_with_no_improvement += 1
        min_final_loss = final_loss

    return env.initial_policy
