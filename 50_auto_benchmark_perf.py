import itertools
import subprocess
import yaml
import os

# Define parameter lists and dependencies to install before running this script
backend_names = [  # mrinufft of course
    # "finufft",  # finufft
    "gpunufft",  # gpuNUFFT, cupy-cuda11x
    "cufinufft",  # cufinufft, cupy-cuda11x
    "tensorflow",  # tensorflow, cupy-cuda11x, tensorflow_nufft, tensorflow_mri => start by tensorflow_mri
    # "torchkbnufft-cpu",  # torchkbnufft, torch
    "torchkbnufft-gpu",  # torchkbnufft, torch
]
trajectories = [
    "./trajs/floret_256x256x176_0.5.bin",
    "./trajs/seiffert_256x256x176_0.5.bin",
    "./trajs/stack_of_spiral_256x256x176_0.5.bin",
    # "./trajs/radial_256x256_0.5.bin",
    # "./trajs/stack2D_of_spiral_256x256_0.5.bin"
]
n_coils_list = [12]

# Read the base configuration file to copy from
with open("./perf/benchmark_config.yaml", "r") as file:
    base_config = yaml.safe_load(file)


combinations = list(itertools.product(backend_names, trajectories, n_coils_list))
benchmark_script = "10_benchmark_perf.py"
os.makedirs("temp_configs", exist_ok=True)

# For each combination, create a temporary configuration file and execute the script
for backend_name, trajectory, n_coils in combinations:
    config = base_config.copy()
    config["backend"]["name"] = backend_name
    config["trajectory"] = trajectory
    config["data"]["n_coils"] = n_coils

    temp_config_path = f"temp_configs"
    temp_config_file = f"config_{backend_name}_{trajectory.split('/')[-1].split('.')[0]}_{n_coils}.yaml"

    complete_file = os.path.join(temp_config_path, temp_config_file)
    with open(complete_file, "w") as file:
        yaml.dump(config, file)

    subprocess.run(
        [
            "python",
            benchmark_script,
            "--config-name",
            temp_config_file,
            "--config-path",
            temp_config_path,
        ]
    )

for file in os.listdir("temp_configs"):
    complete_file = os.path.join("temp_configs", file)
    os.remove(complete_file)
os.removedirs("temp_configs")
