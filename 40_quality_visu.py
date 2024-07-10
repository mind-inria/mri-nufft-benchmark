import numpy as np
import glob
import os
import matplotlib.pyplot as plt

BENCHMARK_DIR = "./outputs-qual"
results_files = glob.glob(BENCHMARK_DIR + "/**/*.npy", recursive=True)

num_files = len(results_files)
print(f"Nombre de fichiers trouvés: {num_files}")

grid_size = int(np.ceil(np.sqrt(num_files)))
fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

for idx, file_path in enumerate(results_files):
    data = np.load(file_path)

    print(f"Dimensions des données pour {file_path}: {data.shape}")

    ax = axs[idx // grid_size, idx % grid_size]

    file_name = os.path.basename(file_path)

    if data.ndim == 3:
        mid_slice = data.shape[0] // 2
        ax.imshow(data[mid_slice], cmap='gray')
        ax.set_title(f'{file_name}\nSlice au milieu ({mid_slice})')
    else:
        ax.imshow(data, cmap='gray')
        ax.set_title(file_name)

    ax.axis('off') 


for idx in range(num_files, grid_size * grid_size):
    fig.delaxes(axs.flatten()[idx])

plt.tight_layout()
plt.show()
