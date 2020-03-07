import random
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with h5py.File('../log/20200306181806.h5', 'r') as f:
    mind = float('inf')
    train_acc_array = f['train_acc'][()]
    train_loss_array = f['train_loss'][()]
    
    # print(np.max(np.sqrt(np.sum(abs(data_full[20])**2, axis=-1))))
    # print(data_full.shape)
    # print(pd.DataFrame(label_full[:1000].reshape(-1, 40)))

    # print(mind)

    # for key in f.keys():
    #     print(f[key])

    acc_len = len(train_acc_array)
    plt.plot([i for i in range(acc_len)], train_acc_array)
    plt.show()

    plt.plot([i for i in range(acc_len)], train_loss_array)
    plt.show()
