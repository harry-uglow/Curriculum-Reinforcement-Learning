import math

import torch

import numpy as np


xl = 0.15
xu = 0.45
yl = -0.35
yu = -0.65


# Normalise coordinates so all are in range [0, 1].
# DEBUG: u/l bounds currently set for waypoint
def normalise_coords(coords, lower=np.array([xl, yl, 0.025]), upper=np.array([xu, yu, 1])):
    return (coords - lower) / (upper - lower)


# Normalise joint angles so -pi -> 0, 0 -> 0.5 and pi -> 1. (mod pi)
def normalise_angles(angles):
    js = angles / np.pi
    rem = lambda x: x - x.astype(int)
    return np.array(
        [rem((j + (np.abs(j) // 2 + 1.5) * 2) / 2.) for j in js])


def train_initial_policy(env):
    x, y = env.get_initial_data()

    p = env.np_random.permutation(len(x))
    env.initial_policy.train_net(x[p], y[p])

    null_action = np.array([0.] * len(y[0]))

    min_final_loss = math.inf
    episodes_with_no_improvement = 0

    # Use DAgger until stabilised
    while episodes_with_no_improvement < 2:
        print("Training initial policy - DAgger iteration " + str(i))
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

        if final_loss < min_final_loss:
            episodes_with_no_improvement = 0
            min_final_loss = final_loss
        else:
            episodes_with_no_improvement += 1

    return env.initial_policy
