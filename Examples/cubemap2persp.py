#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv

Img_path = sys.argv[1]

equiRect = cv2.imread(Img_path)
cv2.namedWindow("cubemap", cv2.WINDOW_NORMAL)
cv2.imshow("cubemap", equiRect)
cv2.waitKey(0)
outShape = [400, 400]
inShape = equiRect.shape[:2]
mapper = fisheyeImgConv()

##############  Uncomment any of the given block to run desired example  ######################

# """
# NOTE : for cubemap2persp conversion you need to always call the cubemap2persp method
FOV = 90
Theta = 0
Phi = 0
Hd = outShape[0]
Wd = outShape[1]
persp = mapper.cubemap2persp(equiRect, FOV, Theta, Phi, Hd, Wd)
cv2.imshow("cubemap", persp)
cv2.waitKey(0)
# """
