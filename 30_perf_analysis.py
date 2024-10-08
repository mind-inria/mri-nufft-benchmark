"""
This script generates benchmark plots from CSV files containing performance metrics.

The generated plots are saved as a PNG file with the specified filename.

Usage:
    python 30_perf_analysis.py <path_to_csv> <traj> <output_filename>
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import glob

sns.set_theme()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Generate benchmark plots and save to specified file."
)
parser.add_argument(
    "path_to_folder_with_csv", type=str, help="Path to the folder with all your csv you want to show."
)
parser.add_argument(
    "traj_dimension", type=int, help="Indicate if your trajectory is in 2D or 3D."
)
parser.add_argument(
    "output_filename", type=str, help="Name of the output file (without extension)."
)
args = parser.parse_args()

# Directory where benchmark result files are stored
BENCHMARK_DIR = args.path_to_folder_with_csv
results_files = glob.glob(BENCHMARK_DIR + "**/*.csv", recursive=True)

# Read and concatenate all CSV files into a single DataFrame
df = pd.concat(map(pd.read_csv, results_files))

# Calculate additional metrics
df["coil_time"] = df["run_time"] / df["n_coils"]
df["coil_mem"] = df["mem_peak"] / df["n_coils"]
df = df.sort_values(["backend"], ascending=False)

tasks = ["forward", "adjoint", "grad"]
metrics = {
    "coil_time": "time (s) /coil",
    "mem_peak": "Peak RAM (GB)",
    "gpu0_mem_GiB_peak": "Peak GPU Mem (GB)",
}

# Remove GPU memory metric if all values are zero
if df["gpu0_mem_GiB_peak"].sum() == 0:
    metrics.pop("gpu0_mem_GiB_peak")
    num_metrics = 2
else:
    num_metrics = 3

# Initialize subplots
fig, axs = plt.subplots(
    3,
    num_metrics,
    sharey=True,
    figsize=(16, 9),
    gridspec_kw=dict(hspace=0.01, wspace=0.05),
)

# Custom palette with specified colors
custom_palette = {1: "black", 12: "darkblue", 32: "purple"}

# Define x-axis limits for each metric, based on the precedent range of values
# first time, then memory, then GPU memory in respective seconds, GB and GB
if num_metrics == 2: # if CPU only
    limits = [(0, 35), (0, 80)] 
elif args.traj_dimension == 2: # if 2D trajectory
    limits = [(0, 0.3), (0, 7), (0, 1)]
else:
    limits = [(0, 1), (0, 50), (0, 20)]

xlims = {k: v for k, v in zip(metrics.keys(), limits)}

# Generate bar plots for each task and metric
for row, task in zip(axs, tasks):
    ddf = df[df["task"] == task]
    for ax, (k) in zip(row[:num_metrics], metrics.keys()):
        sns.barplot(
            ddf,
            x=k,
            y="backend",
            hue="n_coils",
            palette=custom_palette,
            ax=ax,
            errorbar=None,
            width=0.8,
        )
        ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels("")

        max_limit = xlims[k][1]
        for container in ax.containers:
            labels = [
                f"{v:.3f}" if v >= max_limit else "" for v in container.datavalues
            ]
            ax.bar_label(
                container, labels=labels, label_type="center", color="white", fontsize=6
            )


# Set axis labels
for ax, xlabel in zip(axs[-1, :], metrics.values()):
    ax.set_xlabel(xlabel)
for ax, xlabel in zip(axs[0, :], metrics.values()):
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.insert(
        0,
        matplotlib.patches.Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
        ),
    )
    legend_labels.insert(0, "# Coils")
    ax.legend(
        handles, legend_labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0)
    )
    ax.set_title(xlabel, pad=40)

for rl, task in zip(axs[:, 0], tasks):
    rl.set_ylabel(task)

# Rescale x-axis limits and set x-tick labels for each column
for col_ax, xlim in zip(axs.T, xlims.values()):
    for ax in col_ax:
        ax.set_xlim(xlim)
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(xt)}" if xt % 1 == 0 else f"{xt:.1f}" for xt in xticks])

# Save the figure to the specified directory with the provided filename
output_file = BENCHMARK_DIR + f"/{args.output_filename}.png"
plt.savefig(output_file)
plt.show()

# Save the DataFrame to a CSV file
df.to_csv( BENCHMARK_DIR + f"/{args.output_filename}.csv", index=False)