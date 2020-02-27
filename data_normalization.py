#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from math import cos, sin, atan2, sqrt, pi, radians, degrees, ceil

# 数据集路径
data_path = './h5/'
train_file_path = data_path + 'train_data.h5'
test_file_path = data_path + 'test_data.h5'

# 将所有的点集网格尺寸设置为32*32
w = 32
h = 32
# 网格化点集中单个点转化成网格的权重
weight_factor = 0.0235
start_factor = 0.9
end_factor = 1


def normalized_data(data):
    # 创建网格矩阵
    grid_set = np.ones([0, 32, 32, 1], dtype=np.float32)
    # 遍历每个样本
    count = 0
    for sample_data in data:
        # 单个样本网格
        sample_grid = np.ones([32, 32, 1], dtype=np.float32)
        # 遍历样本中的坐标
        for coordinate in sample_data:
            x, y = coordinate
            # 计算样本数据距离左上角坐标的相对距离
            x_distance = abs(x - x_min)
            y_distance = abs(y - y_max)
            # 求出样本数据在网格矩阵的位置角标
            x_index = int(x_distance // grid_interval_x) + 1
            y_index = int(y_distance // grid_interval_y) + 1
            # 如果网格矩阵的值大于权重，则减少相应的权重值
            if sample_grid[x_index][y_index][0] >= weight_factor:
                sample_grid[x_index][y_index][0] -= weight_factor * \
                    (random.random() * abs(end_factor - start_factor) + start_factor)
        count += 1
        print('【计数】：', count)
        print(pd.DataFrame(sample_grid.reshape([w, h])))
        # 样本网格归集
        grid_set = np.append(grid_set, [sample_grid], axis=0)
    return grid_set


# 获取点集数据的网格边界
def get_csv_data_border(data):
    y_max = float("-inf")
    y_min = float("inf")
    x_max = float("-inf")
    x_min = float("inf")
    for coordinate in data:
        x, y = coordinate
        if x > x_max:
            x_max = x
        elif x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        elif y < y_min:
            y_min = y
    return y_max, y_min, x_max, x_min


if __name__ == "__main__":
    with h5py.File(train_file_path, 'r') as f:
        rand_sum_train_data = f['data'][()]
        rand_train_typical_data = f['label'][()]

    with h5py.File(test_file_path, 'r') as f:
        rand_sum_test_data = f['data'][()]
        rand_test_typical_data = f['label'][()]

    # 获取所有点集的最大网格边界
    y_max, y_min, x_max, x_min = get_csv_data_border(
        np.append(rand_sum_train_data, rand_sum_test_data, axis=0).reshape(-1, 2))
    # 获取所有点集的网格宽和高（中心点为坐标原点）
    grid_w = max(ceil(abs(x_max)), ceil(abs(x_min)))*2
    grid_h = max(ceil(abs(y_max)), ceil(abs(y_min)))*2

    # 每个小格子的间距
    grid_interval_x = grid_w/w
    grid_interval_y = grid_h/h

    normalized_train_data = normalized_data(rand_sum_train_data)
    normalized_test_data = normalized_data(rand_sum_test_data)

    if os.access(train_file_path, os.F_OK) == True:
        os.remove(train_file_path)
    open(train_file_path, 'w')
    with h5py.File(train_file_path, 'r+') as f:
        f.create_dataset('data', data=normalized_train_data)
        f.create_dataset('label', data=rand_train_typical_data)

    if os.access(test_file_path, os.F_OK) == True:
        os.remove(test_file_path)
    open(test_file_path, 'w')
    with h5py.File(test_file_path, 'r+') as f:
        f.create_dataset('data', data=normalized_test_data)
        f.create_dataset('label', data=rand_test_typical_data)
