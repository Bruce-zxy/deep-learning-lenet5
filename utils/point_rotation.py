#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
# 绕pointx,pointy逆时针旋转


def Nrotate(angle, valuex, valuey, pointx, pointy):
 valuex = np.array(valuex)
 valuey = np.array(valuey)
 nRotatex = (valuex-pointx)*math.cos(angle) - \
     (valuey-pointy)*math.sin(angle) + pointx
 nRotatey = (valuex-pointx)*math.sin(angle) + \
     (valuey-pointy)*math.cos(angle) + pointy
 return nRotatex, nRotatey
# 绕pointx,pointy顺时针旋转


def Srotate(angle, valuex, valuey, pointx, pointy):
 valuex = np.array(valuex)
 valuey = np.array(valuey)
 sRotatex = (valuex-pointx)*math.cos(angle) + \
     (valuey-pointy)*math.sin(angle) + pointx
 sRotatey = (valuey-pointy)*math.cos(angle) - \
     (valuex-pointx)*math.sin(angle) + pointy
 return sRotatex, sRotatey


import os
import glob
from skimage import io, transform
 #将所有的图片重新设置尺寸为32*32
w = 32
h = 32
c = 1

#mnist数据集中训练数据和测试数据保存地址
train_path = "./mnist/train/"
test_path = "./mnist/test/"

def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index, folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.png'):
            image = io.imread(img)
            image = transform.resize(image, (w, h, c))
            images.append(image)
            labels.append(index)
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


#读取训练数据及测试数据
train_data, train_label = read_image(train_path)
test_data, test_label = read_image(test_path)

for item in train_data[0]:
    print(item)


pointx = 1
pointy = 1
sPointx, sPointy = Nrotate(math.radians(45), pointx, pointy, 0, 0)
print(sPointx, sPointy)
plt.plot([0,pointx], [0,pointy])
plt.plot([0,sPointx], [0,sPointy])
plt.xlim(-3., 3.)
plt.ylim(-3., 3.)
plt.xticks(np.arange(-3., 3., 1))
plt.yticks(np.arange(-3., 3., 1))
plt.show()
