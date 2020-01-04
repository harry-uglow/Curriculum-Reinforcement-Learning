import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# two_stage_baseline_data = [torch.load(f"sparse_dr_{i}M_eval.pt") for i in range(1, 5)]
# curl_data = torch.load(f"curl_eval.pt")
# dense_dr_data = torch.load(f"dense_dr_eval.pt")
from envs.pipelines import pipelines

clrs = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

data = {
    '50p_2cm': [[], [], []],
    '60p_2cm': [[], [], []],
    '70p_2cm': [[], [], []],
    'base': [[], [], []],
}


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(save_base, pipeline, seed):
    tasks = pipeline['curriculum'] + [pipeline['task']]
    datas = [[]]*16

    for task_dir in tasks:
        infiles = glob.glob(os.path.join(f'{save_base}_{seed}_{task_dir}', '*.monitor.csv'))
        for idx, inf in enumerate(infiles):
            with open(inf, 'r') as f:
                f.readline()
                f.readline()
                for num, line in enumerate(f):
                    if num % 19 < 3:
                        continue
                    tmp = line.split(',')
                    if tmp[0][0] == '\x00':
                        tmp = [float(tmp[0][1:])]
                    else:
                        tmp = [float(tmp[0])]
                    datas[idx] = datas[idx] + tmp

    datas = np.array(datas)
    result = []
    for episode in range(len(datas[0])):
        samples = datas[:, episode]
        for idx, sample in enumerate(samples):
            result.append([episode * 16 + idx, sample])

    # if len(result) < bin_size:
    #     return [None, None]

    result = np.array(result)
    x, mu = result[:, 0], result[:, 1]

    # if smooth == 1:
    # x, mu = smooth_reward_curve(x, mu)

    # x, mu = fix_point(x, mu, 64)
    return [x, mu, result[:, 1]]


for key in data:
    seeds = [[], [], []]
    for i in range(3):
        seeds[i] = load_data(f'logs/{key}', pipelines['rack_1'], i * 16)
#         data[key][0] += [i]
#         data[key][1] += [mean]
#         data[key][2] += [std]
    max_len = max([len(res[0]) for res in seeds])
    xs = [x for x in range(0, max_len, 64)]
    mus = []
    errors = []
    for i in range(len(xs)):
        samples = []
        all_samples = []
        for j in range(3):
            # samples += [seeds[j][1][i]]
            to_add = seeds[j][2][i * 64:(i + 1) * 64]
            if len(to_add) > 0:
                all_samples += [to_add]
        all_samples = np.array(all_samples).flatten()
        mus += [np.mean(all_samples)]
        errors += [np.std(all_samples)]
    # xs, mus = smooth_reward_curve(xs, mus)

    # xs, mus = fix_point(xs, mus, 64)
    data[key] = [xs, mus, errors]

fig, ax = plt.subplots()
plt.xlabel('Number of episodes')
plt.ylabel('Reward')

with sns.axes_style("darkgrid"):
    for idx, key in enumerate(data):
        eps, mean, error = data[key]
        mean = np.array(mean)
        error = np.array(error)
        ax.plot(eps, mean, label=f"{key}", linewidth=0.9, color=clrs[idx])
        ax.fill_between(eps, mean - error, mean + error, alpha=0.3, facecolor=clrs[idx])

    plt.ylim(bottom=0)
    # plt.yticks([50, 60, 70, 80, 90])
    plt.legend(loc=2)

    # plt.title("Success rate on Dish Rack over training time")
    # plt.draw()
    plt.show()

    # plt.close(fig)

