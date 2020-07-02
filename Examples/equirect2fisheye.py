#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv

Img_path = sys.argv[1]

equiRect = cv2.imread(Img_path)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",equiRect)
cv2.waitKey(0)
outShape = [250,250]
inShape = equiRect.shape[:2]
mapper = fisheyeImgConv()

##############  Uncomment any of the given block to run desired example  ######################

"""
# For single image Unified Camera model (UCM)
fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,xi=0.5)
cv2.imshow("UCM Model Output",fisheye)
cv2.waitKey(0)
"""

"""
# For single image Extended UCM model
fisheye = mapper.equirect2Fisheye_EUCM(equiRect,outShape=[250,250],f=100,a_=0.4,b_=2,angles=[0,0,0])
cv2.imshow("EUCM Model Output",fisheye)
cv2.waitKey(0)
"""

"""
# For single image Field Of Vide (FOV) model
fisheye = mapper.equirect2Fisheye_FOV(equiRect,outShape=[250,250],f=40,w_=0.5,angles=[0,0,0])
cv2.imshow("FOV model Output",fisheye)
cv2.waitKey(0)
"""

# """
# For single image Double Sphere (DS) model
fisheye = mapper.equirect2Fisheye_DS(equiRect,outShape=[250,250],f=90,a_=0.1,xi_=0.7,angles=[0,0,0])
cv2.imshow("DS Model Output",fisheye)
cv2.waitKey(0)
# """


"""
# For multiple images and the mapping is not changing
fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape)
cv2.imshow("fisheye",fisheye)
cv2.waitKey(0)
fisheye = mapper.applyMap(3,equiRect)
cv2.imshow("fisheye",fisheye)
cv2.waitKey(0)
"""

"""
# For multiple images and mapping is changing
fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape)
cv2.imshow("fisheye",fisheye)
cv2.waitKey(0)

# Changing the distortion coefficient
fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,xi=0.5)
cv2.imshow("fisheye",fisheye)
cv2.waitKey(0)

# Rotate the sphere
fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,angles=[0,0,0])
cv2.imshow("fisheye",fisheye)
cv2.waitKey(0)

"""

"""
# Function performance test
start = time.time()
for i in range(50):
    fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape)
    # cv2.imshow("output",fisheye)
    # cv2.waitKey(0)
print("avg time : ",(time.time()-start)/50.0)
start = time.time()
for i in range(50):
    fisheye = mapper.applyMap(3,equiRect)
print("avg time : ",(time.time()-start)/50.0)
"""