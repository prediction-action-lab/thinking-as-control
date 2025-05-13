
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd

# bases = ['no-mask_auto_pretrained_', 'no-mask_auto_scratch_', 'mask_auto_pretrained_']
bases = ['pretrained_', 'pretrained_use-mask_', 'scratch_use-mask_', 'scratch_']
names = ['Pretrained-Think', 'Pretrained-NoThink', 'Scratch-NoThink', 'Scratch-Think']

results = {}
# plt.figure(figsize=(10, 6))
data_list = []

for base, name in zip(bases, names):

    results[base] = []

    for d in range(5):

        try:
            x = np.load(f"5x5-results/{base}{d}.npy")
        except FileNotFoundError:
            continue
        for t, val in enumerate(x):
            data_list.append({
                'timepoint': t,
                'value': val,
                'series': name,
                'trial': d
            })
        # results[base].append(x)

    # data = np.stack(results[base])
    # # Convert to long-form DataFrame for seaborn
    # df = pd.DataFrame(data)
    # df['trial'] = df.index
    # df_melted = df.melt(id_vars='trial', var_name='timepoint', value_name='value')
    # df_melted['timepoint'] = df_melted['timepoint'].astype(int)
    # sns.lineplot(data=df_melted, x='timepoint', y='value', errorbar='ci', n_boot=1000)

    # yerr = (1.96 / np.sqrt(3)) * np.std(results[base])
    # plt.plot(np.mean(results[base], axis=0))
    # plt.errorbar(x=np.arange(x.size), y=np.mean(results[base], axis=0), yerr=np.std(results[base], axis=0))

# Create DataFrame
df_all = pd.DataFrame(data_list)

# Plot with seaborn
plt.figure(figsize=(10, 8))
sns.lineplot(data=df_all, x='timepoint', y='value', hue='series', errorbar='ci', n_boot=1000)

# Make tick labels larger
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=22, loc=(0.5, 0.125))
# plt.title("Mean Across Trials with Bootstrap CI")
plt.xlabel("Iteration", fontsize=25)
plt.ylabel("Success Rate", fontsize=25)
# plt.xlim([0, 105])
plt.tight_layout()
plt.show()