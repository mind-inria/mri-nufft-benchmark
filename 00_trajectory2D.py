"""
This script generates 2D spiral and radial trajectories for MRI benchmarks using the mrinufft library.
The generated trajectories are saved to files who are in .gitignore

Usage:
    python 00_trajectory2D.py shape_dim1 shape_dim2 --res resolution

Output:
    Files containing the generated 2D trajectories saved in the 'trajs' directory.
"""

import os
import numpy as np
from mrinufft.trajectories import initialize_2D_spiral, initialize_2D_radial
from mrinufft.io import write_trajectory
import argparse


def get_parser():
    """
    Create and return an argument parser for the script.

    """
    parser = argparse.ArgumentParser(
        description="Generate 2D trajectories for the benchmarks."
    )
    parser.add_argument(
        "shape", type=int, nargs=2, default=[192, 192], help="Shape of the 2D image."
    )
    parser.add_argument("--res", type=float, default=0.5, help="Resolution.")
    return parser


if __name__ == "__main__":
    # Create an argument parser and parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Extract 2D shape and field of view (FOV) from arguments
    SHAPE2D = args.shape
    FOV = np.array(SHAPE2D) * args.res

    # Base string for filenames
    base_string = f"{SHAPE2D[0]}x{SHAPE2D[1]}_{args.res}"

    # Generate and save stack of spiral trajectories
    spiral2D = initialize_2D_spiral(Nc=64, Ns=10240, nb_revolutions=7)
    write_trajectory(
        spiral2D,
        FOV,
        SHAPE2D,
        os.path.join("trajs", f"stack2D_of_spiral_{base_string}"),
    )

    # Generate and save radial trajectories
    radial2D = initialize_2D_radial(Nc=64, Ns=10240)
    write_trajectory(
        radial2D, FOV, SHAPE2D, os.path.join("trajs", f"radial_{base_string}")
    )
