#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
sys.path.append("../")
from fisheyeUtils import *

Img_path = sys.argv[1]

cubemap = cv2.imread(Img_path)
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.imshow("Image",cubemap)
cv2.waitKey(0)
outShape = [400,800]
inShape = cubemap.shape[:2]
mapper = fisheyeImgConv()

##############  Uncomment any of the given block to run desired example  ######################

# """
# For single image in horiontal or dice form
# NOTE : The software can automatically determine if the cubemap format is in dice form or horizontal form based on height to width ratio.
start = time.time()

for i in range(20):
	equirect = mapper.cubemap2equirect(cubemap,outShape)
print((time.time()-start)/20)
print("Input shape",cubemap.shape)
print("Output shape",equirect.shape)
cv2.imshow("Image",equirect)
cv2.waitKey(0)
# """

"""
# For multiple images and the mapping is not changing
equirect = mapper.cubemap2equirect(cubemap,outShape)
equirect = mapper.applyMap(2,cubemap)
cv2.imshow("cubemap",equirect)
cv2.waitKey(0)
"""
