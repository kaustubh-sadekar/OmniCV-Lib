#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv

source_path = sys.argv[1]
video = int(sys.argv[2])

if video:
	cap = cv2.VideoCapture(source_path)
	ret, equiRect = cap.read()
else:
	equiRect = cv2.imread(source_path)
	ret = True

cv2.imshow("Input_Image",equiRect)
cv2.waitKey(1)

def nothing(x):
    pass

WINDOW_NAME = "output"
cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME,500,500)

cv2.createTrackbar("focus",WINDOW_NAME,90,150,nothing)
cv2.createTrackbar("distortion",WINDOW_NAME,1,200,nothing)
cv2.createTrackbar("alpha",WINDOW_NAME,180,360,nothing)
cv2.createTrackbar("beta",WINDOW_NAME,180,360,nothing)
cv2.createTrackbar("gamma",WINDOW_NAME,180,360,nothing)

outShape = [500,500]
mapper = fisheyeImgConv()

while ret:
	if video:
		ret,equiRect = cap.read()
		cv2.imshow("Input_Image",equiRect)
		cv2.waitKey(1)
	else:
		ret = True
	f = cv2.getTrackbarPos("focus",WINDOW_NAME)
	dist = cv2.getTrackbarPos("distortion",WINDOW_NAME)/100
	alpha = cv2.getTrackbarPos("alpha",WINDOW_NAME) - 180
	beta = cv2.getTrackbarPos("beta",WINDOW_NAME) - 180
	gamma = cv2.getTrackbarPos("gamma",WINDOW_NAME) - 180

	fisheye = mapper.equirect2Fisheye(equiRect,outShape=outShape,f=f,xi =dist,angles=[alpha,beta,gamma])
	cv2.imshow(WINDOW_NAME,fisheye)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break