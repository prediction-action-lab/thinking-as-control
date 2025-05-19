"""Code to plot RL training in thinking experiments."""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd

bases = ["pretrained_2_", "pretrained_mask_2_", "scratch_", "scratch_nomask_"]
names = ["Pretrained-Think", "Pretrained-NoThink", "Scratch-NoThink", "Scratch-Think"]

results = {}
# plt.figure(figsize=(10, 6))
data_list = []

for base, name in zip(bases, names):

    results[base] = []

    for d in range(5):

        try:
            # x = np.load(f"5x5-results/{base}{d}.npy")
            x = np.load(f"5x5-results/{base}{d}-thinkactions.npy")
        except FileNotFoundError:
            continue
        for t, val in enumerate(x):
            data_list.append({"timepoint": t, "value": val, "series": name, "trial": d})

# Create DataFrame
df_all = pd.DataFrame(data_list)

# Plot with seaborn
plt.figure(figsize=(10, 8))
sns.lineplot(
    data=df_all, x="timepoint", y="value", hue="series", errorbar="ci", n_boot=1000
)

# Make tick labels larger
plt.tick_params(axis="both", labelsize=20)
# plt.legend(fontsize=22, loc=(0.5, 0.125))
plt.legend(fontsize=20, ncols=2)
# plt.title("Mean Across Trials with Bootstrap CI")
plt.xlabel("Iteration", fontsize=25)
plt.ylabel("Success Rate", fontsize=25)
# plt.ylabel("Fraction Time Thinking", fontsize=25)
# plt.xlim([0, 105])
# plt.ylim([-0.199, 1.1])
# plt.ylim([-0.05, 0.6])
plt.tight_layout()
plt.show()
