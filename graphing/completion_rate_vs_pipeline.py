import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# two_stage_baseline_data = [torch.load(f"sparse_dr_{i}M_eval.pt") for i in range(1, 5)]
# curl_data = torch.load(f"curl_eval.pt")
# dense_dr_data = torch.load(f"dense_dr_eval.pt")
clrs = sns.color_palette("husl", 5)

data = {
    '1cm': [[], [], []],
    '2cm': [[], [], []],
    '4cm': [[], [], []],
    '8cm': [[], [], []],
    '16cm': [[], [], []],
}


for i in range(50, 100, 10):
    for key in data:
        try:
            results = torch.load(f'train_lengths/{i}p_{key}_wall_train_lengths.pt')
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
        mean = np.array(mean)
        error = np.array(error)
        ax.plot(x, mean, label=f"{key} pipeline", linewidth=0.9, color=clrs[i])
        ax.fill_between(x, mean - error, mean + error, alpha=0.3, facecolor=clrs[i])
    ax.legend(loc=2)


    # plt.xlim((0, 4000000))
    plt.xticks([50, 60, 70, 80, 90])

    # plt.title("Success rate on Dish Rack over training time")
    plt.draw()
    plt.show()

    # plt.savefig(f'imgs/{game}.png')
    # plt.close(fig)

