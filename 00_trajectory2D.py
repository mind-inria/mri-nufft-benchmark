"""
Generate trajectories for the benchmarks.
"""

import os

import numpy as np
from mrinufft.trajectories import initialize_2D_spiral, initialize_2D_radial
from mrinufft.io import write_trajectory


import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Generate trajectories for the benchmarks.")
    parser.add_argument("shape", type=int, nargs=2, default=[192,192], help="Shape of the 2D image.")
    parser.add_argument("--res", type=float, default=0.5, help="Resolution.")
    return parser


if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args() 
    args = argparse.Namespace(shape=[256, 256], res=0.5)
    print(args)
    SHAPE2D = args.shape
    FOV = np.array(SHAPE2D) * args.res

    base_string = f"{SHAPE2D[0]}x{SHAPE2D[1]}_{args.res}"

    # Stack of Spiral
    spiral2D = initialize_2D_spiral(Nc=64, Ns=10240, nb_revolutions=7)
    write_trajectory(spiral2D, FOV, SHAPE2D, os.path.join("trajs", f"stack2D_of_spiral_{base_string}"))

    # Radial
    radial2D = initialize_2D_radial(Nc=64, Ns=10240)
    write_trajectory(radial2D, FOV, SHAPE2D, os.path.join("trajs", f"radial_{base_string}"))
