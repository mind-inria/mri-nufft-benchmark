from brainweb_dl import get_mri
import numpy as np 

samples = get_mri(sub_id=5, shape=(256,256,176))

np.save('cpx_cartesian.npy', samples)