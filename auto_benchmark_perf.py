"""
This script generates benchmark configurations for different MRI reconstruction backends and trajectories, 
executes the benchmarking script, and cleans up temporary configuration files.

Usage:
    1. Ensure that all necessary backends are installed and accessible.
    2. Run 'python auto_benchmark_perf.py'

The script performs the following tasks:
    - Reads a base configuration file (`benchmark_config.yaml`) that defines default settings.
    - Generates all combinations of backend names, trajectories, and number of coils specified.
    - For each combination, it creates a temporary YAML configuration file.
    - Calls the benchmark script with the generated configuration.
    - Cleans up by deleting all temporary configuration files after execution.

Note:
    The benchmark script should be designed to accept a configuration file path as an argument.
"""

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

    temp_config_path = "temp_configs"
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

# Clean up temporary configuration files
for file in os.listdir("temp_configs"):
    complete_file = os.path.join("temp_configs", file)
    os.remove(complete_file)
os.removedirs("temp_configs")
