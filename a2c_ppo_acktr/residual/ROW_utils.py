import torch

import numpy as np

from a2c_ppo_acktr.residual.initial_policy_model import InitialPolicy, train_nn

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


class ReachOverWallInitialPolicy(object):

    def __init__(self, num_inputs, num_outputs, hidden_size=50):
        self.waypoint_net = InitialPolicy(num_inputs, num_outputs, hidden_size=hidden_size)
        self.target_net = InitialPolicy(num_inputs, num_outputs, hidden_size=hidden_size)

    def train(self, target, train_x, train_y):
        if target:
            self.target_net = train_nn(self.target_net, train_x, train_y)
        else:
            self.waypoint_net = train_nn(self.waypoint_net, train_x, train_y)

    def act(self, end_pose, ip_input):
        with torch.no_grad():
            end_x = end_pose[0]
            end_z = end_pose[2]

            net = self.waypoint_net if end_x < -0.005 and end_z < 0.445 - end_x else self.target_net
            return net(ip_input)


def solve_ik(env, group, distance):
    _, path, _, _ = env.call_lua_function('solve_ik', strings=[group, distance])
    num_path_points = len(path) // len(env.joint_handles)
    path = np.reshape(path, (num_path_points, len(env.joint_handles)))
    distances = np.array([path[i + 1] - path[i]
                          for i in range(0, len(path) - 1)])
    velocities = distances * 20  # Distances should be covered in 0.05s
    return path, velocities


def get_demo_path(env):
    path, velocities_WP = solve_ik(env, 'IK_GroupW', 'tip_waypoint')
    path_to_WP = path[:-1]

    env.call_lua_function('set_joint_angles', ints=env.init_config_tree, floats=path[-1])
    path_to_trg, velocities_trg = solve_ik(env, 'IK_GroupT', 'tip_target')

    return normalise_angles(path_to_WP), normalise_angles(path_to_trg[:-1]), \
        velocities_WP, velocities_trg


def train_initial_policy(env):
    x_WP, x_trg, y_WP, y_trg = get_demo_path(env)
    xs = {'T': x_trg, 'W': x_WP}
    ys = {'T': y_trg, 'W': y_WP}

    env.initial_policy = ReachOverWallInitialPolicy(len(x_WP[0]), len(y_WP[0]))
    env.initial_policy.train(False, xs['W'], ys['W'])
    env.initial_policy.train(True, xs['T'], ys['T'])

    null_action = np.array([0.] * len(x_WP[0]))
    # Use DAgger for 8 episodes
    for i in range(8):
        print("Training initial policy - DAgger iteration " + str(i))
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(null_action)
            end_x = env.end_pose[0]
            end_z = env.end_pose[2]
            if end_x < -0.005 and end_z < 0.445 - end_x:
                goal = 'W'
                distance_name = 'tip_waypoint'
            else:
                goal = 'T'
                distance_name = 'tip_target'
            new_x, new_y = solve_ik(env, 'IK_Group' + goal, distance_name)
            if len(new_x) != 0:
                xs[goal] = np.append(xs[goal], normalise_angles(new_x[:-1]), axis=0)
                ys[goal] = np.append(ys[goal], new_y, axis=0)

        p = env.np_random.permutation(len(xs['W']))
        env.initial_policy.train(False, xs['W'][p], ys['W'][p])
        p = env.np_random.permutation(len(xs['T']))
        env.initial_policy.train(True, xs['T'][p], ys['T'][p])

    return env.initial_policy


def setup_ROW_Env(seed, rank):
    from a2c_ppo_acktr.ReachOverWallEnv import ReachOverWallEnv
    env = ReachOverWallEnv(seed, rank)
    ip = train_initial_policy(env)
    env.close()
    return ip
