import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import glob
from collections import defaultdict

sns.set_theme()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate benchmark plots and save to specified file.')
parser.add_argument('output_filename', type=str, help='Name of the output file (without extension).')
args = parser.parse_args()

# Replace with correct directory
BENCHMARK_DIR = "./outputs"
results_files = glob.glob(BENCHMARK_DIR + "/**/**/*.csv", recursive=True)
# print(results_files)

# You can specify the path directly (more specific or for only one file)
# results_files = ["file_path.csv"]

df = pd.concat(map(pd.read_csv, results_files))
df["coil_time"] = df["run_time"] / df["n_coils"]
df["coil_mem"] = df["mem_peak"] / df["n_coils"]
df = df.sort_values(["backend"], ascending=False)


fig, axs = plt.subplots(
    3, 3, sharey=True, figsize=(16, 9), gridspec_kw=dict(hspace=0.01, wspace=0.05)
)
tasks = ["forward", "adjoint", "grad"]
metrics = {
    "coil_time": "time (s) /coil",
    "mem_peak": "Peak RAM (GB)",
    "gpu0_mem_GiB_peak": "Peak GPU Mem (GB)",
}
palette = {k: v for k, v in zip(metrics.keys(), ["magma", "rocket", "mako"])}

# Calculate maximum values for each metric
max_coil_time = df["coil_time"].max()
max_mem_peak = df["mem_peak"].max()
max_gpu_mem_peak = df["gpu0_mem_GiB_peak"].max()

# Define x-axis limits for each metric
# xlims = {k: v for k, v in zip(metrics.keys(), [(0, 10), (0, 80), (0, 8)])}
xlims = {
    "coil_time": (0, max_coil_time),
    "mem_peak": (0, max_mem_peak),
    "gpu0_mem_GiB_peak": (0, max_gpu_mem_peak),
}

# Generate bar plots for each task and metric
for row, task in zip(axs, tasks):
    ddf = df[df["task"] == task]
    for ax, (k, p) in zip(row, palette.items()):
        sns.barplot(
            ddf,
            x=k,
            y="backend",
            hue="n_coils",
            palette=p,
            ax=ax,
            errorbar=None,
            width=0.8,
        )
        ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels("")


# Labels
for ax, xlabel in zip(axs[-1, :], metrics.values()):
    ax.set_xlabel(xlabel)
for ax, xlabel in zip(axs[0, :], metrics.values()):
    h, l = ax.get_legend_handles_labels()
    h.insert(
        0,
        matplotlib.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
        ),
    )
    l.insert(0, "# Coils")
    ax.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(xlabel, pad=40)

for rl, task in zip(axs[:, 0], tasks):
    rl.set_ylabel(task)

# Rescale x-axis limits and set x-tick labels for each column
for col_ax, xlim in zip(axs.T, xlims.values()):
    for ax in col_ax:
        ax.set_xlim(xlim)
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{xt:.0f}" for xt in xticks])

# Save the figure to the specified directory with the provided filename
output_file = f"./outputs/{args.output_filename}.png"
plt.savefig(output_file)
plt.show()