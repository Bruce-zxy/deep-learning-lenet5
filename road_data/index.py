#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import xlrd
import random
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, atan2, sqrt, pi, radians, degrees, ceil, isnan

data_dict = [
    ['秦园中路A.xlsx', 129],
    ['龟山南路B.xlsx', 260],
    ['沿江大道B.xlsx', 43],
    ['八一路B.xlsx', 129],
    ['八一路2A.xlsx', 128],
    ['拦江路B.xlsx', 71],
    ['中山路A.xlsx', 95],
    ['八一路2B.xlsx', 128],
    ['中山路B.xlsx', 95],
    ['解放路2B.xlsx', 16],
    ['沿江大道A.xlsx', 43],
    ['龟山南路A.xlsx', 260],
    ['张之洞路A.xlsx', 274],
    ['解放路B.xlsx', 19],
    ['解放路2A.xlsx', 16],
    ['解放路A.xlsx', 19],
    ['拦江路A.xlsx', 71],
    ['张之洞路B.xlsx', 274],
    ['秦园中路B.xlsx', 129],
    ['八一路A.xlsx', 129]
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


if __name__ == "__main__":

    height_set = []

    for file_name, angle in data_dict:
        y_max = float("-inf")
        y_min = float("inf")
        rotated_data = np.empty([0, 2], dtype=np.float32)
        original_data = excel2np("./" + file_name)
        centroid_x, centroid_y = get_centroid(original_data)
        for index, coordinate in enumerate(original_data):
            x, y = coordinate
            if y_max < y:
                y_max = y
            elif y_min > y:
                y_min = y
            n_x, n_y = n_rotate(radians(angle), x, y, centroid_x, centroid_y)
            rotated_data = np.append(rotated_data, [[n_x,  n_y]], axis=0)
        print(original_data.shape)
        print(file_name)
        print('旋转角度：', angle)
        height_set.append(y_max-y_min)

        xs, ys = original_data.T
        nxs, nys = rotated_data.T
        plt.scatter(nxs, nys, c=randomcolor(), s=1)
        plt.scatter(xs, ys, c=randomcolor(), s=1)
        plt.show()
    
    print(height_set)
    print(sum(height_set))
    
