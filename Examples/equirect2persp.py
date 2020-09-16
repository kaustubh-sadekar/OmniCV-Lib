#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv

Img_path = sys.argv[1]

equiRect = cv2.imread(Img_path)

outShape = [400, 400]
inShape = equiRect.shape[:2]
mapper = fisheyeImgConv()

##############  Uncomment any of the given block to run desired example  ######################

# """
# For single image
FOV = 90
Theta = 0
Phi = 0
Hd = outShape[0]
Wd = outShape[1]

start = time.time()
for i in range(20):
    persp = mapper.eqruirect2persp(equiRect, FOV, Theta, Phi, Hd, Wd)
print((time.time() - start) / 20)
print("Input shape", equiRect.shape)
print("Output shape", persp.shape)
cv2.imshow("perspective", persp)
cv2.imshow("equirect", equiRect)
cv2.waitKey(0)
# """

"""
# If the mapping is not changing you can use applyMap method else you need to call the getPerspective method with changing parameters
FOV = 90
Theta = 0
Phi = 0
Hd = outShape[0]
Wd = outShape[1]
persp = mapper.eqruirect2persp(equiRect,FOV,Theta,Phi,Hd,Wd)
persp = mapper.applyMap(4,equiRect)
cv2.imshow("cubemap",persp)
cv2.waitKey(0)
"""
