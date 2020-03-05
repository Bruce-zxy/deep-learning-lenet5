#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import xlrd
import random
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from math import cos, sin, atan2, sqrt, pi, radians, degrees, ceil, isnan, floor

# 将所有的点集网格尺寸设置为32*32
w = 32
h = 32
# 网格化点集中单个点转化成网格的权重
weight_factor = 0.0325

# data_dict = [
#     ['解放路B.xlsx', 19, '0'],
#     ['龟山南路B.xlsx', 260, '1'],
#     ['中山路B.xlsx', 95, '2']
# ]

data_dict = [
    ['解放路A.xlsx', 19, '0'],
    ['解放路B.xlsx', 19, '0'],
    ['解放路2A.xlsx', 16, '0'],
    ['解放路2B.xlsx', 16, '0'],
    ['龟山南路A.xlsx', 260, '1'],
    ['龟山南路B.xlsx', 260, '1'],
    ['拦江路B.xlsx', 71, '1'],
    ['拦江路A.xlsx', 71, '1'],
    ['沿江大道A.xlsx', 43, '1'],
    ['沿江大道B.xlsx', 43, '1'],
    ['秦园中路A.xlsx', 129, '1'],
    ['秦园中路B.xlsx', 129, '1'],
    ['八一路A.xlsx', 129, '2'],
    ['八一路B.xlsx', 129, '2'],
    ['八一路2A.xlsx', 128, '2'],
    ['八一路2B.xlsx', 128, '2'],
    ['张之洞路A.xlsx', 274, '2'],
    ['张之洞路B.xlsx', 274, '2'],
    ['中山路A.xlsx', 95, '2'],
    ['中山路B.xlsx', 95, '2']
 ]


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7',
                '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#"+color

# 获取二维点集的中心点坐标
def get_centroid(point_set):
    c_x, c_y = zip(*point_set)
    centroid_x = sum(c_x)/len(c_x)
    centroid_y = sum(c_y)/len(c_y)
    return centroid_x, centroid_y

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

# 逆时针旋转坐标点
def n_rotate(angle, valuex, valuey, centerx, centery):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-centerx)*cos(angle) - \
        (valuey-centery)*sin(angle) + centerx
    nRotatey = (valuex-centerx)*sin(angle) + \
        (valuey-centery)*cos(angle) + centery
    return nRotatex, nRotatey

def excel2np(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    datamatrix = np.zeros((nrows, 2))
    for x in range(2):
        cols = table.col_values(x)
    #     minVals = min(cols)
    #     maxVals = max(cols)
    #     cols1 = np.matrix(cols)  # 把list转换为矩阵进行矩阵操作
    #     ranges = maxVals - minVals
    #     b = cols1 - minVals
    #     normcols = b / ranges  # 数据进行归一化处理
        datamatrix[:, x] = cols  # 把数据进行存储
    return datamatrix

def normalized_data(data_dict):
    # 创建网格矩阵
    grid_set = np.ones([0, 32, 32, 1], dtype=np.float32)
    grid_type_set = np.ones([0, 1], dtype=np.int32)

    for dtype, values in data_dict.items():
        for sample_data in values:
            # 单个样本网格
            sample_grid = np.ones([32, 32, 1], dtype=np.float32)
            for x, y in sample_data:
                # 计算样本数据距离左上角坐标的相对距离
                x_distance = abs(x - x_min)
                y_distance = abs(y - y_max)
                # 求出样本数据在网格矩阵的位置角标
                x_index = int(x_distance // grid_interval_x)
                y_index = int(y_distance // grid_interval_y)
                # 如果网格矩阵的值大于权重，则减少相应的权重值
                if sample_grid[x_index][y_index][0] >= weight_factor:
                    sample_grid[x_index][y_index][0] -= weight_factor
            # 样本网格归集
            grid_set = np.append(grid_set, [sample_grid], axis=0)
            grid_type_set = np.append(grid_type_set, [[int(dtype)]], axis=0)
    return grid_set, grid_type_set


if __name__ == "__main__":

    # 样本高度
    init = 4
    # 样本平移量
    interval = 0.8
    height_set = {}
    sample_len_set = {}
    point_dict = {
        '0': [],
        '1': [],
        '2': []
    }
    point_set = np.empty([0, 2], dtype=np.float32)
    point_type_set = np.empty([0], dtype=np.str)

    # 旋转道路，并找到道路在y轴上的高度和样本总数
    for file_name, angle, data_type in data_dict:
        name, ext = file_name.split('.')
        y_max = float("-inf")
        y_min = float("inf")
        rotated_data = np.empty([0, 2], dtype=np.float32)
        # 读取道路Excel文件
        original_data = excel2np("./" + file_name)
        # 获取道路点集的重心
        centroid_x, centroid_y = get_centroid(original_data)
        # 旋转道路
        for index, coordinate in enumerate(original_data):
            x, y = coordinate
            if y_max < y:
                y_max = y
            elif y_min > y:
                y_min = y
            n_x, n_y = n_rotate(radians(angle), x, y, centroid_x, centroid_y)
            rotated_data = np.append(rotated_data, [[n_x, n_y]], axis=0)

        print('【' + name + '旋转前总点数】：', len(original_data))
        print('【' + name + '旋转后总点数】：', len(rotated_data))
        print('旋转角度：', angle)
        print('----------------------------')

        # 道路y轴高度
        height = y_max-y_min
        height_set[name] = (height)
        # 道路样本总数
        num = floor((height - init)/interval) + 1
        sample_len_set[name] = (num)

        # 切分道路为若干个样本块
        slicing_data = [[] for i in range(num)]
        base_y = y_min + (height - init - (num - 1)*interval)/2
        # 按预估样本数量遍历
        for i in range(num):
            start_y = base_y + i * interval
            end_y = start_y + init
            # 遍历旋转后的道路点集
            for x, y in rotated_data:
                if y >= start_y and y < end_y:
                    slicing_data[i].append([x, y])

        point_dict[data_type] += slicing_data

        # xs, ys = original_data.T
        # nxs, nys = rotated_data.T
        # plt.scatter(nxs, nys, c=randomcolor(), s=1)
        # plt.scatter(xs, ys, c=randomcolor(), s=1)
        # plt.show()
    
    print('----------------------------')
    print('【预估样本数量】：', sum([values for key, values in sample_len_set.items()]))
    print('【世纪样本总计】：', sum([len(values) for key, values in point_dict.items()]))
    print('【样本】：', json.dumps(sample_len_set, ensure_ascii=False, indent=2))
    print('【路高】：', json.dumps(height_set, ensure_ascii=False, indent=2))
    
    for dtype, points_type in point_dict.items():
        for points in points_type:
            centroid_x, centroid_y = get_centroid(points)
            for index, point in enumerate(points):
                x,y = point
                new_point = [x-centroid_x, y-centroid_y]
                # 点集坐标中心化
                points[index] = new_point
                point_set = np.append(point_set, [new_point], axis=0)
                point_type_set = np.append(point_type_set, [dtype], axis=0)

    y_max, y_min, x_max, x_min = get_csv_data_border(point_set)
    # 获取所有点集的网格宽和高（中心点为坐标原点）
    grid_w = max(abs(x_max), abs(x_min))*2
    grid_h = max(abs(y_max), abs(y_min))*2

    # 每个小格子的间距
    grid_interval_x = grid_w/w
    grid_interval_y = grid_h/h

    grid_set, grid_type_set = normalized_data(point_dict)

    # # 网格化所有样本
    # for dtype, points_type in point_dict.items():
    #     for points in points_type:
    #             new_points = np.asarray(points)
    #             print(pd.DataFrame(new_points))
    #             xs, ys = new_points.T
    #             plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], ls="-", lw=1, c=randomcolor(), label="line")
    #             plt.scatter(xs, ys, c=randomcolor(), s=1)
    #             plt.show()

    if os.access('./data.h5', os.F_OK) == True:
        os.remove('./data.h5')
    open('./data.h5', 'w')
    with h5py.File('./data.h5', 'r+') as f:
        f.create_dataset('data', data=grid_set)
        f.create_dataset('label', data=grid_type_set)

    print(grid_set.shape)
    print(grid_type_set.shape)
    print(y_max, y_min, x_max, x_min)
    
