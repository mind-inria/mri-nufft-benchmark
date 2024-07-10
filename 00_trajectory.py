"""
Generate trajectories for the benchmarks.
"""

import os

import numpy as np
from mrinufft.trajectories import initialize_3D_floret, initialize_2D_spiral, stack, initialize_3D_seiffert_spiral
from mrinufft.io import write_trajectory


import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Generate trajectories for the benchmarks.")
    parser.add_argument("shape", type=int, nargs=3, default=[192,192,128], help="Shape of the 3D image.")
    parser.add_argument("--res", type=float, default=0.5, help="Resolution.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    SHAPE3D = args.shape
    FOV = np.array(SHAPE3D) * args.res

    base_string = f"{SHAPE3D[0]}x{SHAPE3D[1]}x{SHAPE3D[2]}_{args.res}"
    # Stack of Spiral
    stack_of_spiral = stack(initialize_2D_spiral(Nc=64, Ns=10240, nb_revolutions=7),nb_stacks=208)
    write_trajectory(stack_of_spiral, FOV, SHAPE3D, os.path.join("trajs", f"stack_of_spiral_{base_string}"))

    # Floret

    floret = initialize_3D_floret(Nc=3994//6, Ns=10240, nb_revolutions=6)
    write_trajectory(floret, FOV, SHAPE3D, os.path.join("trajs", f"floret_{base_string}"))


    # Seiffert Spiral
    seiffert = initialize_3D_seiffert_spiral(Nc=3994//6, Ns=10240, nb_revolutions=6)

    write_trajectory(seiffert, FOV, SHAPE3D, os.path.join("trajs", f"seiffert_{base_string}"))
