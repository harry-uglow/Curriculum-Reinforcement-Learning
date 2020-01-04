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
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

data = {
    '50': [[], [], []],
    '60': [[], [], []],
    '70': [[], [], []],
    '80': [[], [], []],
    '90': [[], [], []],
    '100': [[], [], []],
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


def load_data(save_base, pipeline, smooth=1, bin_size=100):
    tasks = pipeline['curriculum'] + [pipeline['task']]
    datas = [[]]*16

    for task_dir in tasks:
        infiles = glob.glob(os.path.join(f'{save_base}_0_{task_dir}', '*.monitor.csv'))
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
        result.append([episode * 16, np.mean(samples), np.std(samples)])

    # if len(result) < bin_size:
    #     return [None, None]

    result = np.array(result)
    x, mu, std = result[:, 0], result[:, 1], result[:, 2]

    # if smooth == 1:
    #     x, mu = smooth_reward_curve(x, mu)
    #
    # x, mu = fix_point(x, mu, bin_size)
    return [x, mu, std]


for tsr in range(50, 100, 100):
    eps, mean, error = load_data(f'logs/{tsr}p_1cm', pipelines['rack_1'])
#         episodes = [x / 64 for x in results[0]]
#         mean = np.mean(episodes)
#         std = np.std(episodes)
#         data[key][0] += [i]
#         data[key][1] += [mean]
#         data[key][2] += [std]

fig, ax = plt.subplots()
plt.xlabel('Number of episodes')
plt.ylabel('Reward')

with sns.axes_style("darkgrid"):
    # for tsr, key in enumerate(data):
    ax.plot(eps, mean, label=f"1cm pipeline", linewidth=0.9, color=clrs[0])
    ax.fill_between(eps, mean - error, mean + error, alpha=0.3, facecolor=clrs[0])
    ax.legend(loc=2)

    plt.ylim((0, np.max(mean + error)))
    # plt.xticks([50, 60, 70, 80, 90])

    # plt.title("Success rate on Dish Rack over training time")
    # plt.draw()
    plt.show()

    # plt.close(fig)

