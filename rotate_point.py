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


pointx = 1
pointy = 1
sPointx, sPointy = Nrotate(math.radians(45), pointx, pointy, 0, 0)
print(sPointx, sPointy)
plt.plot([0, pointx], [0, pointy])
plt.plot([0, sPointx], [0, sPointy])
plt.xlim(-3., 3.)
plt.ylim(-3., 3.)
plt.xticks(np.arange(-3., 3., 1))
plt.yticks(np.arange(-3., 3., 1))
plt.show()
