"""Code to produce heatmaps used in policy iteration experiment."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_heatmap(all_data, policy_data=None):

    n_plots = len(all_data.keys())

    # Set up the vertical layout
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 8), sharex=True)

    # Determine common color scale
    vmin = min([all_data[key].min() for key in all_data])
    vmax = min([all_data[key].max() for key in all_data])
    # vmax = max(data1.max(), data2.max(), data3.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "coolwarm"

    # Titles
    titles = ["Heatmap 1", "Heatmap 2", "Heatmap 3"]

    # Plot each heatmap
    for ax, title in zip(axes, all_data.keys()):

        data = all_data[title]
        policy = policy_data[title]
        print(policy)

        sns.heatmap(
            data, ax=ax, cmap=cmap, norm=norm, cbar=False, annot=False, square=True
        )
        ax.set_title(title, fontsize=25)

        # Add arrow annotations to each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                action = policy[i, j]
                if action == "U":
                    # ax.arrow(j, i + 0.15, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
                    ax.arrow(
                        j + 0.5,
                        i + 0.75,
                        0.0,
                        -0.3,
                        head_width=0.1,
                        head_length=0.1,
                        fc="white",
                        ec="white",
                    )
                elif action == "D":
                    ax.arrow(
                        j,
                        i - 0.15,
                        0,
                        0.3,
                        head_width=0.1,
                        head_length=0.1,
                        fc="black",
                        ec="black",
                    )
                elif action == "L":
                    # ax.arrow(j + 0.15, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
                    ax.arrow(
                        j + 0.75,
                        i + 0.5,
                        -0.3,
                        0,
                        head_width=0.1,
                        head_length=0.1,
                        fc="white",
                        ec="white",
                    )
                elif action == "R":
                    ax.arrow(
                        j + 0.25,
                        i + 0.5,
                        0.3,
                        0,
                        head_width=0.1,
                        head_length=0.1,
                        fc="white",
                        ec="white",
                    )
                    # ax.text(j + 0.5, i + 0.5, '→', ha='center', va='center', color='white', fontsize=12)

    # Create a single colorbar that spans all three heatmaps
    # Use ScalarMappable for a shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Dummy array for colorbar

    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        cmap="coolwarm",
    )
    cbar.set_label("Value Scale", fontsize=25)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for colorbar
    plt.show()


# Generate dummy 2x10 data for each heatmap
# data1 = np.random.rand(2, 10)
# data2 = np.random.rand(2, 10)
# data3 = np.random.rand(2, 10)

# data = {}
# data['Heatmap 1'] = data1
# data['Heatmap 2'] = data2
# data['Heatmap 3'] = data3

# plot_heatmap(data)
