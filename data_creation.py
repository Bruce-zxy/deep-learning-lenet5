#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from math import cos, sin, atan2, sqrt, pi, radians, degrees, ceil, isnan
from skimage import io, transform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

TRAIN_CSV_PATH = './pointdata4/traindata/'
TEST_CSV_PATH = './pointdata4/testdata/'

data_path = './h5/'
train_file_path = data_path + 'initial_train_data.h5'
test_file_path = data_path + 'initial_test_data.h5'

# 按旋转角度分类的子级目录
label_dirs = [[16, 19], [43,71,129, 260], [95,128,129, 274]]
# 按道路分类的父级目录
label_set = [0, 1, 2]

# 获取二维点集的中心点坐标
def get_centroid(point_set):
    c_x, c_y = zip(*point_set)
    centroid_x = sum(c_x)/len(c_x)
    centroid_y = sum(c_y)/len(c_y)
    return centroid_x, centroid_y

# 逆时针旋转坐标点


def n_rotate(angle, valuex, valuey, centerx, centery):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-centerx)*cos(angle) - \
        (valuey-centery)*sin(angle) + centerx
    nRotatey = (valuex-centerx)*sin(angle) + \
        (valuey-centery)*cos(angle) + centery
    return nRotatex, nRotatey

# 获取csv文件的列表


def get_csv_list(path):
    csv_file_list = []
    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.endswith('csv'):
            csv_file_list.append(path + "/" + file_name)
    return csv_file_list

# 获取csv文件中的点集数据


def get_csv_data(path_list):
    # 创建空的定维数组
    sum_data = np.empty([0, 1024, 2], dtype=np.float32)

    # 遍历每个csv文件
    for path in path_list:
        # 将每个csv文件读取为Numpy的数据
        data = np.genfromtxt(path, delimiter=',', dtype=np.float32)[:, :2]
        data_len = len(data)
        empty_len = 1024 - data_len

        # 完整的1024个元数据=csv文件数据+在csv文件中随机指定下标数据
        count = 0
        while count < empty_len:
            data = np.append(
                data, [data[random.randint(0, data_len-1)]], axis=0)
            count += 1
        sum_data = np.append(sum_data, [data], axis=0)
    print(sum_data.shape)
    return sum_data


# 随机打乱点集数据
def exchange_data_index(sum_data, label_data):
    cursor_index = 0
    max_range = len(sum_data)
    while cursor_index < max_range:
        random_index = random.randint(0, max_range-1)
        temp_sum_data = sum_data[0]
        temp_label_data = label_data[0]

        sum_data = np.delete(sum_data, 0, axis=0)
        label_data = np.delete(label_data, 0, axis=0)
        sum_data = np.insert(sum_data, random_index, temp_sum_data, axis=0)
        label_data = np.insert(label_data, random_index,
                               temp_label_data, axis=0)

        cursor_index += 1
    return sum_data, label_data


def get_label_and_data(root_path, label_dirs):
    sum_data = np.empty([0, 1024, 2], dtype=np.float32)
    typical_data = np.empty([0], dtype=np.int32)

    for data_type, label_dir_set in enumerate(label_dirs):
        print(">> 现在进入【第%d类】数据" % (data_type+1))
        for rotate_angle in label_dir_set:
            print("-- 需要旋转%d度的数据集：" % (rotate_angle))
            # 获取csv文件列表
            csv_list = get_csv_list(
                root_path + str(data_type) + '/' + str(rotate_angle))
            # 获取csv文件点集数据
            csv_data = get_csv_data(csv_list)
            # 遍历样本数据
            for i, sample_data in enumerate(csv_data):
                # 求出点集的中心坐标点
                centroid_x, centroid_y = get_centroid(sample_data)
                # 根据中心坐标点旋转点集中的点
                
                for index, coordinate in enumerate(sample_data):
                    x, y = coordinate
                    n_x, n_y = n_rotate(
                        radians(rotate_angle), x, y, centroid_x, centroid_y)
                    # 旋转后的点集坐标中心化
                    sample_data[index] = [n_x-centroid_x, n_y-centroid_y]
                # 旋转后的点集回归原列表
                csv_data[i] = sample_data
                # 归集点集标签
                typical_data = np.append(typical_data, [data_type], axis=0)
            # 将每个不同数量的样本合并到主列表中（n,1024,2）=>（m,n,1024,2）
            sum_data = np.append(sum_data, csv_data, axis=0)

    return sum_data, typical_data


if __name__ == "__main__":

    sum_train_data, train_typical_data = get_label_and_data(
        TRAIN_CSV_PATH, label_dirs)
    sum_test_data, test_typical_data = get_label_and_data(
        TEST_CSV_PATH, label_dirs)

    # 随机打乱点集数据
    rand_sum_train_data, rand_train_typical_data = exchange_data_index(
        sum_train_data, train_typical_data)
    rand_sum_test_data, rand_test_typical_data = exchange_data_index(
        sum_test_data, test_typical_data)

    if os.access(data_path, os.F_OK) == False:
        os.mkdir(data_path)

    if os.access(train_file_path, os.F_OK) == True:
        os.remove(train_file_path)
    open(train_file_path, 'w')
    with h5py.File(train_file_path, 'r+') as f:
        f.create_dataset('data', data=rand_sum_train_data)
        f.create_dataset('label', data=rand_train_typical_data)

    if os.access(test_file_path, os.F_OK) == True:
        os.remove(test_file_path)
    open(test_file_path, 'w')
    with h5py.File(test_file_path, 'r+') as f:
        f.create_dataset('data', data=rand_sum_test_data)
        f.create_dataset('label', data=rand_test_typical_data)
