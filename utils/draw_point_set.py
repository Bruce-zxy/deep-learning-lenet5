#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# 数据集路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = base_dir + '/h5/'
train_file_path = data_path + 'initial_train_data.h5'
test_file_path = data_path + 'initial_test_data.h5'

def get_csv_data_border(data):
    y_max = float("-inf")
    y_min = float("inf")
    x_max = float("-inf")
    x_min = float("inf")
    for coordinate in data:
        x, y = coordinate
        if x == 0 and y == 0:
            continue
        elif x > x_max:
            x_max = x
        elif x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        elif y < y_min:
            y_min = y
    return y_max, y_min, x_max, x_min

with h5py.File(train_file_path, 'r') as f:
    rand_sum_train_data = f['data'][()]
    rand_train_typical_data = f['label'][()]

with h5py.File(test_file_path, 'r') as f:
    rand_sum_test_data = f['data'][()]
    rand_test_typical_data = f['label'][()]

y_max, y_min, x_max, x_min = get_csv_data_border(rand_sum_test_data.reshape(-1, 2))
print(y_max, y_min, x_max, x_min)
for data in rand_sum_test_data:
    xs, ys = data.T
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], c="#FDA400")
    plt.scatter(xs, ys, s=1, c='#0000FF', alpha=0.4, label='类别A')
    plt.show()
