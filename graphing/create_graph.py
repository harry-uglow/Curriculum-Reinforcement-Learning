import torch
import matplotlib.pyplot as plt

two_stage_baseline_data = [torch.load(f"sparse_dr_{i}M_eval.pt") for i in range(1, 5)]
curl_data = torch.load(f"curl_eval.pt")
dense_dr_data = torch.load(f"dense_dr_eval.pt")

for idx, data in enumerate(two_stage_baseline_data):
    xs, _ = data
    two_stage_baseline_data[idx][0] = [x + (idx + 1) * 1000000 for x in xs]

fig = plt.figure()
# for idx, data in enumerate(two_stage_baseline_data):
#     x, y = data
#     plt.plot(x, y, label=f"{idx + 1} Million dense timesteps", linewidth=0.9)
x, y = curl_data
plt.plot(x, y, label=f"CuRL Progress", linewidth=2.)
x, y = dense_dr_data
plt.plot(x[:-1], y[:-1], label=f"Dense Reward Baseline", linewidth=2.)

plt.xlabel('Total number of training timesteps')
plt.ylabel('Evaluation')

plt.xlim((0, 4000000))
plt.ylim((0, 100))

plt.title("Success rate on Dish Rack over training time")
plt.legend(loc=4)
plt.show()
plt.draw()

plt.close(fig)