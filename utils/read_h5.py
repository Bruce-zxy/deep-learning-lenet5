import random
import h5py
import numpy as np

with h5py.File('../road_data/data.h5', 'r') as f:
    data_full = f['data'][()]
    print(np.max(np.sqrt(np.sum(abs(data_full[20])**2, axis=-1))))
    print(data_full.shape)

    for key in f.keys():
        print(f[key])
