#!/usr/bin/env/python

import cv2
import numpy as np
import math
import time
import sys
import os
from omnicv import fisheyeImgConv

img_path = "./input/equirect.jpg"
def test_inputPath1():
    assert os.path.isfile(img_path),"Wrong path... Run the tests.py code after changing the directory inside test folder"

param_file_path = "./input/fisheyeParams.txt"
def test_parameterFilePath():
    assert os.path.isfile(param_file_path),"fisheyeParams.txt file not present"

mapper = fisheyeImgConv(param_file_path)
equiRect = cv2.imread(img_path)

# [1] Testing equirect2Fisheye_UCM method
def test_equirect2Fisheye_UCM():
    inShape = equiRect.shape[:2]
    fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=[250,250],xi=0.5)
    assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"
    
# [2] Testing equirect2Fisheye_EUCM method
def test_equirect2Fisheye_EUCM():
    fisheye = mapper.equirect2Fisheye_EUCM(equiRect,outShape=[250,250],f=100,a_=0.4,b_=2,angles=[0,0,0])
    assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

# [3] Testing equirect2Fisheye_FOV method
def test_equirect2Fisheye_FOV():
    fisheye = mapper.equirect2Fisheye_FOV(equiRect,outShape=[250,250],f=40,w_=0.5,angles=[0,0,0])
    assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

# [4] Testing equirect2Fisheye_DS method
def test_equirect2Fisheye_DS():
    fisheye = mapper.equirect2Fisheye_DS(equiRect,outShape=[250,250],f=90,a_=0.4,xi_=0.8,angles=[0,0,0])
    assert fisheye.shape[:2]==(250,250),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/fisheye.jpg"
def test_inputPath2():
    assert os.path.isfile(img_path),"fisheye.jpg image not present"

fisheye = cv2.imread(img_path)
#inShape = fisheye.shape[:2]

# [5] Testing fisheye2equirect method for mode=1
def test_fisheye2equirect1():
    equiRect = mapper.fisheye2equirect(fisheye,[200,400])
    assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"

# [6] Testing fisheye2equirect method for mode=2
def test_fisheye2equirect2():
    equiRect = mapper.fisheye2equirect(fisheye,[200,400])
    cubemap = mapper.equirect2cubemap(equiRect,modif=True)
    equiRect = mapper.cubemap2equirect(cubemap,[200,400])
    assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/equirect.jpg"
def test_inputPath3():
    assert os.path.isfile(img_path),"equirect.jpg image not present"

equiRect = cv2.imread(img_path)
#inShape = equiRect.shape[:2]

# [7] Testing equirect2cubemap method for horizontal mode
def test_equirect2cubemap1():
    cubemap = mapper.equirect2cubemap(equiRect,side=256)
    assert cubemap.shape[:2]==(256,1536),"Output dimensions for generated fisheye image do not match !"

# [8] Testing equirect2cubemap method for dice mode
def test_equirect2cubemapDice():
    cubemap = mapper.equirect2cubemap(equiRect,side=256,dice=True)
    assert cubemap.shape[:2]==(768,1024),"Output dimensions for generated fisheye image do not match !"

# [9] Testing eqruirect2persp method
def test_equirect2persp():
    persp = mapper.eqruirect2persp(equiRect,90,0,0,400,400)
    assert persp.shape[:2]==(400,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/cubemap.jpg"
def test_inputPath4():
    assert os.path.isfile(img_path),"cubemap.jpg image not present"

cubemap = cv2.imread(img_path)
#inShape = cubemap.shape[:2]
#outShape = [200,400]

# [10] Testing cubemap2equirect method
def test_cubemap2equirect():
    equiRect = mapper.cubemap2equirect(cubemap,[200,400])
    assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/cubemap_dice.jpg"
def test_inputPath5():
    assert os.path.isfile(img_path),"cubemap_dice.jpg image not present"

cubemap = cv2.imread(img_path)
#inShape = cubemap.shape[:2]
#outShape = [200,400]

# [11] Testing cubemap2equirect method for horizontal mode
def test_cubemap2equirect():
    equiRect = mapper.cubemap2equirect(cubemap,[200,400])
    assert equiRect.shape[:2]==(200,400),"Output dimensions for generated fisheye image do not match !"


img_path = "./input/cubemap.jpg"
def test_inputPath6():
    assert os.path.isfile(img_path),"cubemap.jpg image not present"
cubemap = cv2.imread(img_path)
#inShape = cubemap.shape[:2]

# [12] Testing cubemap2persp method for horizontal mode
def test_cubemap2persp():
    persp = mapper.cubemap2persp(cubemap,90,0,0,400,400)
    assert persp.shape[:2]==(400,400),"Output dimensions for generated fisheye image do not match !"

img_path = "./input/cubemap_dice.jpg"
def test_inputPath7():
    assert os.path.isfile(img_path),"cubemap_dice.jpg image not present"

cubemap = cv2.imread(img_path)
#inShape = cubemap.shape[:2]

# [13] Testing cubemap2persp method for dice mode
def test_cubemap2persp():
    persp = mapper.cubemap2persp(cubemap,90,0,0,400,400)
    assert persp.shape[:2]==(400,400),"Output dimensions for generated fisheye image do not match !"

