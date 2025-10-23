"""Code to plot RL training in thinking experiments."""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd


inexact_subgoals = False
plot_actions = False

if inexact_subgoals:
    bases = ["pretrained_", "pretrained_mask_", "scratch_mask_", "scratch_"]
else:
    bases = ["pretrained_2_", "pretrained_mask_2_", "scratch_", "scratch_nomask_"]
names = ["Pretrained-Think", "Pretrained-NoThink", "Scratch-NoThink", "Scratch-Think"][:len(bases)]

results = {}
# plt.figure(figsize=(10, 6))
data_list = []

for base, name in zip(bases, names):

    results[base] = []

    for d in range(10):

        try:
            if plot_actions:
                if inexact_subgoals:
                    x = np.load(f"inexact-results/{base}{d}-thinkactions.npy")
                else:
                    x = np.load(f"cr-5x5-results/{base}{d}-thinkactions.npy")
            else:
                if inexact_subgoals:
                    x = np.load(f"inexact-results/{base}{d}.npy")
                else:
                    x = np.load(f"cr-5x5-results/{base}{d}.npy")
        except FileNotFoundError:
            continue
        for t, val in enumerate(x):
            data_list.append({"timepoint": t, "value": val, "series": name, "trial": d, 'run': d})

# Create DataFrame
df_all = pd.DataFrame(data_list)

# Plot with seaborn
plt.figure(figsize=(10, 8))
sns.lineplot(
    data=df_all, x="timepoint", y="value", hue="series", estimator='mean', lw=3, errorbar="ci", n_boot=1000, #fill_kwargs={"alpha": 0.3}
)
sns.lineplot(
    data=df_all, x="timepoint", y="value", hue="series", units='run', lw=1, estimator=None, alpha=0.3,
)

handles, labels = plt.gca().get_legend_handles_labels()
handles = handles[:4]
labels = labels[:4]

# Make tick labels larger
plt.tick_params(axis="both", labelsize=25)
# plt.legend(fontsize=22, loc=(0.5, 0.125))
plt.legend(handles, labels, fontsize=20, ncols=2, loc=(0.02, 0.01))
# plt.legend(fontsize=20, ncols=2, loc=3)
# plt.title("Mean Across Trials with Bootstrap CI")
plt.xlabel("Iteration", fontsize=35)
if plot_actions:
    plt.ylabel("Fraction Time Thinking", fontsize=25)
    plt.ylim([-0.05, 0.6])
    # if inexact_subgoals:
    #     plt.legend(handles, labels, fontsize=20, loc=(0.1, 0.125))
    # else:
    plt.legend(handles, labels, fontsize=20, ncols=2, loc=1)
else:
    plt.ylabel("Success Rate", fontsize=35)
    plt.ylim([-0.199, 1.1])
plt.xlim([0, 105])

plt.tight_layout()
plt.show()
