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
    "output_filename", type=str, help="Name of the output file (without extension)."
)
args = parser.parse_args()

# Replace with correct directory
BENCHMARK_DIR = "./outputs"
results_files = glob.glob(BENCHMARK_DIR + "/**/**/*.csv", recursive=True)

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

# Custom palette with specified colors
custom_palette = {1: "black", 12: "darkblue", 32: "purple"}


# Define x-axis limits for each metric
xlims = {k: v for k, v in zip(metrics.keys(), [(0, 5), (0, 80), (0, 20)])}

# Generate bar plots for each task and metric
for row, task in zip(axs, tasks):
    ddf = df[df["task"] == task]
    for ax, (k) in zip(row, metrics.keys()):
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
                f"{v:.1f}" if v >= max_limit else "" for v in container.datavalues
            ]
            ax.bar_label(container, labels=labels, label_type="center", color="white", fontsize=6)


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
    ax.set_xticklabels([f"{xt:.1f}" for xt in xticks])

# Save the figure to the specified directory with the provided filename
output_file = BENCHMARK_DIR + f"/{args.output_filename}.png"
plt.savefig(output_file)
plt.show()
