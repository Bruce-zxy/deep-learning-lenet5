#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, atan2, sqrt, pi, radians, degrees

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 按旋转角度分类的子级目录
label_dirs = [[16, 19], [43, 260], [129, 274]]
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
    nRotatex = (valuex-centerx)*cos(angle) - (valuey-centery)*sin(angle) + centerx
    nRotatey = (valuex-centerx)*sin(angle) + (valuey-centery)*cos(angle) + centery
    return nRotatex, nRotatey

# 绘制点集
def draw_data_set(data):
    y_max, y_min, x_max, x_min = get_csv_data_border(data)

    d2_point_set = data.reshape(-1, 2)
    # centroid_x, centroid_y = get_centroid(d2_point_set)
    x_set, y_set = d2_point_set.T

    plt.plot([x_min, x_min, x_max, x_max, x_min],
             [y_min, y_max, y_max, y_min, y_min])
    plt.xlim(x_min-10, x_max+10)
    plt.ylim(y_min-10, y_max+10)
    plt.scatter(x_set, y_set, s=1)
    # plt.scatter(centroid_x, centroid_y, c="#FF0000", s=5)
    plt.show()

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
    sum_data = np.empty([0, 1024, 2], dtype=np.dtype('<f4'))

    # 遍历每个csv文件
    for path in path_list:
        # 将每个csv文件读取为Numpy的数据
        data = np.genfromtxt(path, delimiter=',', dtype=np.dtype('<f4'))[:, :2]
        data_len = len(data)
        empty_len = 1024 - data_len

        # 完整的1024个元数据=csv文件数据+在csv文件中随机指定下标数据
        count = 0
        while count < empty_len:
            data = np.append(
                data, [data[random.randint(0, data_len-1)]], axis=0)
            count += 1
        sum_data = np.append(sum_data, [data], axis=0)
    print("--", sum_data.shape)
    return sum_data


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
    # print(y_max - y_min)
    # print(x_max - x_min)
    return y_max, y_min, x_max, x_min


if __name__ == "__main__":

    TRAIN_CSV_PATH = './pointdata2/traindata2/'
    TEST_CSV_PATH = './pointdata2/testdata2/'

    sum_data = np.empty([0, 1024, 2], dtype=np.dtype('<f4'))
    typical_data = [[[], []], [[], []], [[], []]]

    # original_train_data_set = []
    # original_test_data_set = []
    for data_type, label_dir_set in enumerate(label_dirs):
        print("现在进入【第%d类】数据" % (data_type+1))
        # temp_train_data_set = []
        # temp_test_data_set = []
        for sub_index, rotate_angle in enumerate(label_dir_set):
            print("-- 需要旋转%d度的数据集：" % (rotate_angle))
            # 获取csv文件列表
            csv_list = get_csv_list(TRAIN_CSV_PATH + str(data_type) + '/' + str(rotate_angle))
            # 获取csv文件点集数据
            csv_data = get_csv_data(csv_list)

            # 遍历样本数据
            for i,sample_data in enumerate(csv_data):
                print(sample_data.shape)
                # 求出点集的中心坐标点
                centroid_x, centroid_y = get_centroid(sample_data)
                print("中心坐标点为：(%.10f, %.10f)" % (centroid_x, centroid_y))

                draw_data_set(sample_data)

                # 根据中心坐标点旋转点集中的点
                for index, coordinate in enumerate(sample_data):
                    x, y = coordinate
                    sample_data[index] = n_rotate(radians(rotate_angle), x, y, centroid_x, centroid_y)
                # 旋转后点集回归原列表
                csv_data[i] = sample_data
                
                draw_data_set(sample_data)

            sum_data = np.append(sum_data, csv_data, axis=0)
            typical_data[data_type][sub_index].append(csv_data.tolist())

            # csv_data = grid_csv_data(csv_data)
            # csv_data = grid_csv_data(csv_data)

            # temp_train_data_set.append(get_csv_list(
            #     TRAIN_CSV_PATH + str(data_type) + '/' + str(rotate_angle)))
            # temp_test_data_set.append(get_csv_list(
            #     TEST_CSV_PATH + str(data_type) + '/' + str(rotate_angle)))

        # original_train_data_set.append(temp_train_data_set)
        # original_test_data_set.append(temp_test_data_set)

    # print(np.asarray(original_train_data_set).shape)
    # print(original_test_data_set)
