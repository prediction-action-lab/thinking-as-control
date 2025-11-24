"""Code to plot RL training in thinking experiments."""
import sys
import argparse
import os
import re
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pretrained PyTorch model with specified options."
    )

    parser.add_argument(
        "--results_directory",
        type=str,
        default=None,
        help="Path to the pretrained PyTorch model file (.pt or .pth)",
    )

    parser.add_argument(
        "--plot_actions",
        action="store_true",
        default=False,
        help='If set, mask out "thinking" actions during evaluation',
    )

    return parser.parse_args()


args = parse_args()

results_directory = args.results_directory
plot_actions = args.plot_actions


# Optional: set a set of labels to use for plot
# base_to_label = {
#     'pretrained': 'Pretrained-Think',
#     'pretrained_mask': 'Pretrained-NoThink',
#     'scratch': 'Scratch-Think',
#     'scratch_mask': 'Scratch-NoThink'
# }
base_to_label = {}

# names = ["Pretrained-Think", "Pretrained-NoThink", "Scratch-NoThink", "Scratch-Think"][:len(bases)]

data_list = []

for filename in os.listdir(results_directory):

    if ".npy" not in filename:
        continue

    if plot_actions and "-thinkactions.npy" not in filename:
        continue

    if not plot_actions and "-thinkactions.npy" in filename:
        continue

    base = "_".join(filename.split("_")[:-1])
    seed = re.findall(r"-?\d+", filename)[-1]

    if base in base_to_label:
        base = base_to_label[base]

    x = np.load(os.path.join(results_directory, filename))

    for t, val in enumerate(x):
        data_list.append({"timepoint": t, "value": val, "series": base, "run": seed})

# Create DataFrame
df_all = pd.DataFrame(data_list)

# Plot with seaborn
plt.figure(figsize=(10, 8))
sns.lineplot(
    data=df_all,
    x="timepoint",
    y="value",
    hue="series",
    estimator="mean",
    lw=3,
    errorbar="ci",
    n_boot=1000,  # fill_kwargs={"alpha": 0.3}
)
sns.lineplot(
    data=df_all,
    x="timepoint",
    y="value",
    hue="series",
    units="run",
    lw=1,
    estimator=None,
    alpha=0.3,
)

handles, labels = plt.gca().get_legend_handles_labels()
handles = handles[:4]
labels = labels[:4]

# Make tick labels larger
plt.tick_params(axis="both", labelsize=25)
plt.legend(handles, labels, fontsize=20, ncols=2, loc=(0.02, 0.01))

plt.xlabel("Iteration", fontsize=35)
if plot_actions:
    plt.ylabel("Fraction Time Thinking", fontsize=25)
    plt.ylim([-0.05, 0.6])
    plt.legend(handles, labels, fontsize=20, ncols=2, loc=1)
else:
    plt.ylabel("Success Rate", fontsize=35)
    plt.ylim([-0.199, 1.1])
plt.xlim([0, 105])

plt.tight_layout()
plt.show()
