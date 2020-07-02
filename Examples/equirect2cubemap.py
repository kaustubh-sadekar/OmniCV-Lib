#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
sys.path.append('../')
from fisheyeUtils import *


img_path = sys.argv[1]
dice = int(sys.argv[2])

equiRect = cv2.imread(img_path)

inShape = equiRect.shape[:2]
mapper = fisheyeImgConv()


##############  Uncomment any of the given block to run desired example  ######################

"""
# For single image in horiontal form
cubemap = mapper.equirect2cubemap(equiRect,side=256,modif=0,dice=dice)
cv2.namedWindow("cubemap",cv2.WINDOW_NORMAL)
# cv2.imwrite("../data/cubemap.jpg",cubemap)
cv2.imshow("cubemap",cubemap)
cv2.waitKey(0)
"""

# """
# For single image in dice form
start = time.time()
for i in range(20):
	cubemap = mapper.equirect2cubemap(equiRect,side=256,dice=dice)
print((time.time()-start)/20)
print("Input shape",equiRect.shape)
print("Output shape",cubemap.shape)
# cv2.imwrite("cubemap.jpg",cubemap)
cv2.imshow("cubemap",cubemap)
cv2.imshow("Equirect",equiRect)
cv2.waitKey(0)
# """


"""
# For multiple images and the mapping is not changing
fisheye = mapper.equirect2cubemap(equiRect,side=256,dice=True)
fisheye = mapper.applyMap(1,equiRect)
cv2.imshow("fisheye",fisheye)
cv2.waitKey(0)
"""
