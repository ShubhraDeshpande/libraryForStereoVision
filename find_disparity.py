"""ref : https://docs.opencv.org/trunk/d9/dba/classcv_1_1StereoBM.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html"""

import cv2
import numpy as np
img1 = cv2.imread("tsucuba_left.png",0)

img2 = cv2.imread("tsucuba_right.png",0)

stereo = cv2.StereoBM_create(numDisparities=80, blockSize=11)
# blur1 = cv2.blur(img1,(3,3))
# blur2 = cv2.blur(img2,(3,3))
disparity = stereo.compute(img1,img2)
disparity_smooth = cv2.blur(disparity,(5,5))

cv2.imwrite('task2_disparity.jpg',disparity_smooth)
#cv2.imshow('gray.jpg',disparity)