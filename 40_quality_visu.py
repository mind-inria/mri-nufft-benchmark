"""
This script loads and visualizes MRI reconstruction results from multiple NumPy files.

The script will:
    - Find all `.npy` files in the specified directory and its subdirectories.
    - Print the number of files found and their dimensions.
    - Display the contents of each file in a grid of subplots. If the data is 3-dimensional, the middle slice will be displayed.
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Directory where benchmark result files are stored
BENCHMARK_DIR = "./outputs-qual"
results_files = glob.glob(BENCHMARK_DIR + "/**/*.npy", recursive=True)

# Print the number of files found
num_files = len(results_files)
print(f"Nombre de fichiers trouvés: {num_files}")

# Determine grid size for plotting
grid_size = int(np.ceil(np.sqrt(num_files)))
fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

# Load and display each file
for idx, file_path in enumerate(results_files):
    data = np.load(file_path)

    print(f"Dimensions des données pour {file_path}: {data.shape}")

    ax = axs[idx // grid_size, idx % grid_size]

    file_name = os.path.basename(file_path)

    if data.ndim == 3:
        mid_slice = data.shape[0] // 2
        ax.imshow(data[mid_slice], cmap="gray")
        ax.set_title(f"{file_name}\nSlice au milieu ({mid_slice})")
    else:
        ax.imshow(data, cmap="gray")
        ax.set_title(file_name)

    ax.axis("off")

# Remove empty subplots if any
for idx in range(num_files, grid_size * grid_size):
    fig.delaxes(axs.flatten()[idx])

plt.tight_layout()
plt.show()
