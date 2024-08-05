"""
This script downloads MRI data using the brainweb_dl library and saves it as a NumPy array.

Output:
    cpx_cartesian.npy: A file containing the MRI data in NumPy array format.
"""

from brainweb_dl import get_mri
import numpy as np

samples = get_mri(sub_id=5, shape=(256, 256, 176))

np.save("cpx_cartesian.npy", samples)
