#!/usr/bin/env/python
import cv2
import numpy as np
import math
import sys
from omnicv import fisheyeImgConv


def nothing(x):
    pass

WINDOW_NAME = "image"
Image_Path = sys.argv[1]
RADIUS = None

# Example of using the converter class
frame = cv2.imread(Image_Path)
cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME,800,800)
N = 2*max(frame.shape)
cv2.createTrackbar('radius',WINDOW_NAME,0,N,nothing)
cv2.createTrackbar('Cx',WINDOW_NAME,N//2,N,nothing)
cv2.createTrackbar('Cy',WINDOW_NAME,N//2,N,nothing)

while True:
    if True:
        frame = cv2.imread(Image_Path)
        radius = cv2.getTrackbarPos('radius',WINDOW_NAME)
        Cx = cv2.getTrackbarPos('Cx',WINDOW_NAME)
        Cy = cv2.getTrackbarPos('Cy',WINDOW_NAME)
        frame = cv2.circle(frame,(Cx,Cy),radius,(0,200,0),2)
        cv2.imshow(WINDOW_NAME,frame)
        if cv2.waitKey(1)&0xFF == 27:
            RADIUS = radius
            cv2.destroyAllWindows()
            break


WINDOW_NAME = "set aperture"

cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME,600,1200)
cv2.createTrackbar('aperture',WINDOW_NAME,0,1000,nothing)
cv2.createTrackbar('del Cx',WINDOW_NAME,500,1000,nothing)
cv2.createTrackbar('del Cy',WINDOW_NAME,500,1000,nothing)

# Example of using the converter class
frame = cv2.imread(Image_Path)
frame = cv2.circle(frame,(frame.shape[1]//2,frame.shape[0]//2),4,(255,255,255),-1)
frame = cv2.circle(frame,(frame.shape[1]//2,frame.shape[0]//2),2,(0,0,0),-1)

outShape = [400,800]
inShape = frame.shape[:2]

mapper = fisheyeImgConv()
mapper.fisheye2equirect(frame,outShape,edit_mode=True)
while True:
    if True:
        aperture = cv2.getTrackbarPos('aperture',WINDOW_NAME)
        delx = cv2.getTrackbarPos('del Cx',WINDOW_NAME) -500
        dely = cv2.getTrackbarPos('del Cy',WINDOW_NAME) -500
        frame2 = mapper.fisheye2equirect(frame,outShape,aperture=aperture,delx=delx,dely=dely,radius=RADIUS,edit_mode=True)
        frame2 = cv2.line(frame2,(int(frame2.shape[1]*0.25),0),(int(frame2.shape[1]*0.25),int(frame2.shape[0])),(0,180,0),1)
        frame2 = cv2.line(frame2,(int(frame2.shape[1]*0.75),0),(int(frame2.shape[1]*0.75),int(frame2.shape[0])),(0,180,0),1)
        frame2 = cv2.line(frame2,(0,int(frame2.shape[0]*0.5)),(int(frame2.shape[1]),int(frame2.shape[0]*0.5)),(0,180,0),1)
        frame2 = cv2.line(frame2,(int(frame2.shape[1]*0.5),0),(int(frame2.shape[1]*0.5),int(frame2.shape[0])),(0,180,0),1)
        cv2.imshow(WINDOW_NAME,frame2)
        if cv2.waitKey(1)&0xFF == 27:
            print("aperture : ",aperture)
            print("delx : ",delx)
            print("dely : ",dely)
            f = open("../fisheyeParams.txt","w+")
            f.write(str(RADIUS)+"\n")
            f.write(str(aperture)+"\n")
            f.write(str(delx)+"\n")
            f.write(str(dely)+"\n")
            f.close()
            break
        frame2[:,:,:] = 0
        cv2.imshow("image",frame2)