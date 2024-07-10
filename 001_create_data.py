from brainweb_dl import get_mri
import numpy as np 

samples = get_mri(sub_id=5, shape=(176, 256,256))

np.save('cpx_cartesian2D.npy', samples)