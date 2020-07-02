#!/usr/bin/env/python

import cv2
import numpy as np
import math
import time
import sys
import os
from omnicv import fisheyeImgConv

mapper = fisheyeImgConv()

print("\n\n########################################################\n")
print("Running tests for python library ....")


img_path = "./input/equirect.jpg"
assert os.path.isfile(img_path),"Wrong path... Run the tests.py code after changing the directory inside test folder"

equiRect = cv2.imread(img_path)
outShape = [250,250]
inShape = equiRect.shape[:2]

print("[1] Testing equirect2Fisheye_UCM method ...")
fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,xi=0.5)
img_path = "./outputs/UCM_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

print("[2] Testing equirect2Fisheye_EUCM method ...")
fisheye = mapper.equirect2Fisheye_EUCM(equiRect,outShape=[250,250],f=100,a_=0.4,b_=2,angles=[0,0,0])
img_path = "./outputs/EUCM_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

print("[3] Testing equirect2Fisheye_FOV method ...")
fisheye = mapper.equirect2Fisheye_FOV(equiRect,outShape=[250,250],f=40,w_=0.5,angles=[0,0,0])
img_path = "./outputs/FOV_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

print("[4] Testing equirect2Fisheye_DS method ...")
fisheye = mapper.equirect2Fisheye_DS(equiRect,outShape=[250,250],f=90,a_=0.4,xi_=0.8,angles=[0,0,0])
img_path = "./outputs/DS_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

param_file_path = "./input/fisheyeParams.txt"
assert os.path.isfile(param_file_path),"fisheyeParams.txt file not present"

img_path = "./input/fisheye.jpg"
assert os.path.isfile(img_path),"fisheye.jpg image not present"

fisheye = cv2.imread(img_path)
outShape = [200,400]
inShape = fisheye.shape[:2]

print("[5] Testing fisheye2equirect method for mode=1...")
mapper = fisheyeImgConv(param_file_path)
equiRect = mapper.fisheye2equirect(fisheye,outShape)
img_path = "./outputs/f2e_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"

# Mode 2 
print("[6] Testing fisheye2equirect method for mode=2...")
equiRect = mapper.fisheye2equirect(fisheye,outShape)
cubemap = mapper.equirect2cubemap(equiRect,modif=True)
equiRect = mapper.cubemap2equirect(cubemap,outShape)
img_path = "./outputs/f2e_out_mode2.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"


img_path = "./input/equirect.jpg"
equiRect = cv2.imread(img_path)
inShape = equiRect.shape[:2]

print("[7] Testing equirect2cubemap method for horizontal mode...")
cubemap = mapper.equirect2cubemap(equiRect,side=256)
img_path = "./outputs/e2c_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert cubemap.shape[:2]==(256,1536),"Output dimensions for generated fisheye image do not match !"

print("[8] Testing equirect2cubemap method for dice mode...")
cubemap = mapper.equirect2cubemap(equiRect,side=256,dice=True)
img_path = "./outputs/e2c_out_dice.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert cubemap.shape[:2]==(768,1024),"Output dimensions for generated fisheye image do not match !"

print("[9] Testing eqruirect2persp method ...")
persp = mapper.eqruirect2persp(equiRect,90,0,0,400,400)
img_path = "./outputs/e2p_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert persp.shape[:2]==(400,400),"Output dimensions for generated fisheye image do not match !"


img_path = "./input/cubemap.jpg"
assert os.path.isfile(img_path),"cubemap.jpg image not present"
cubemap = cv2.imread(img_path)
inShape = cubemap.shape[:2]
outShape = [200,400]

print("[10] Testing cubemap2equirect method ...")
equiRect = mapper.cubemap2equirect(cubemap,outShape)
img_path = "./outputs/c2e_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/cubemap_dice.jpg"
assert os.path.isfile(img_path),"cubemap_dice.jpg image not present"
cubemap = cv2.imread(img_path)
inShape = cubemap.shape[:2]
outShape = [200,400]

print("[11] Testing cubemap2equirect method for horizontal mode...")
equiRect = mapper.cubemap2equirect(cubemap,outShape)
img_path = "./outputs/c2e_out_dice.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/cubemap.jpg"
assert os.path.isfile(img_path),"cubemap.jpg image not present"
cubemap = cv2.imread(img_path)
inShape = cubemap.shape[:2]

print("[12] Testing cubemap2persp method for horizontal mode...")
persp = mapper.cubemap2persp(cubemap,90,0,0,400,400)
img_path = "./outputs/c2p_out.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert persp.shape[:2]==(400,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/cubemap_dice.jpg"
assert os.path.isfile(img_path),"cubemap_dice.jpg image not present"
cubemap = cv2.imread(img_path)
inShape = cubemap.shape[:2]

print("[13] Testing cubemap2persp method for dice mode...")
persp = mapper.cubemap2persp(cubemap,90,0,0,400,400)
img_path = "./outputs/c2p_out_dice.jpg"
assert os.path.isfile(img_path),"Output reference image not present"
assert persp.shape[:2]==(400,400),"Output dimensions for generated fisheye image do not match !"

print("All tests completed ...")