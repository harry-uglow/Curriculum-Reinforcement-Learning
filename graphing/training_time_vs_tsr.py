import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# two_stage_baseline_data = [torch.load(f"sparse_dr_{i}M_eval.pt") for i in range(1, 5)]
# curl_data = torch.load(f"curl_eval.pt")
# dense_dr_data = torch.load(f"dense_dr_eval.pt")
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
    '1cm': [[], [], []],
    '2cm': [[], [], []],
    '4cm': [[], [], []],
    '8cm': [[], [], []],
    '16cm': [[], [], []],
}


for i in range(50, 101, 10):
    for key in data:
        if key != '1cm' and (i == 90 or i == 100):
            continue
        try:
            results = torch.load(f'train_lengths/{i}p_{key}_rack_train_lengths.pt')
        except FileNotFoundError:
            continue
        episodes = [x / 64 for x in results[0]]
        mean = np.mean(episodes)
        std = np.std(episodes)
        data[key][0] += [i]
        data[key][1] += [mean]
        data[key][2] += [std]

fig, ax = plt.subplots()
plt.xlabel('Target success rate (%)')
plt.ylabel('Number of episodes')

with sns.axes_style("darkgrid"):
    for i, key in enumerate(data):
        x, mean, error = data[key]
        str_mean = [f"${mu}$" for mu in mean]
        print(" & ".join(str_mean))
        mean = np.array(mean)
        error = np.array(error)
        ax.plot(x, mean, label=f"{key} pipeline", linewidth=0.9, color=clrs[i])
        ax.fill_between(x, mean - error, mean + error, alpha=0.3, facecolor=clrs[i])
    ax.legend(loc=2)


    # plt.xlim((0, 4000000))
    plt.xticks([50, 60, 70, 80, 90, 100])

    # plt.title("Success rate on Dish Rack over training time")
    plt.draw()
    plt.show()

    # plt.savefig(f'imgs/{game}.png')
    # plt.close(fig)

