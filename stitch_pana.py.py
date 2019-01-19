
"""code of task1_1 contains code for all the task1 subtasks. The code is inspired by built in functions of opencv 
and the parameters are tuned"""

"""ref: 
sift : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
knn : https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
homography : https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
image warping task: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

"""


import cv2
import numpy as np
print("\n starting task1_1")
img1 = cv2.imread("mountain1.jpg")
gray1 = cv2.imread("mountain1.jpg",0)
img2 = cv2.imread("mountain2.jpg")
gray2 = cv2.imread("mountain2.jpg",0)
# gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)
cv2.drawKeypoints(gray1,kp1,img1)
cv2.drawKeypoints(gray2,kp2,img2)

cv2.imwrite("task1_sift1.jpg",img1)
cv2.imwrite("task1_sift2.jpg",img2)

MIN_MATCH_COUNT = 10
print("_________________________________________________________________________")
print("\n starting task1_2")

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2,outImg=None)
cv2.imwrite("task1 matches knn.jpg",img3)
print("_________________________________________________________________________")
print("\n starting task1_3")

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    print(M)

print("_________________________________________________________________________")
print("\n starting task1_4")

matchesMask = np.array(matchesMask)
src=[]
for i in range(matchesMask.shape[0]):
	if matchesMask[i] == 1:
		src.append(good[i])
src = src[:10]

img7 = cv2.drawMatches(gray1,kp1,gray2,kp2,src,flags=2,outImg=None)

cv2.imwrite("task1_matches.jpg",img7)

print("_________________________________________________________________________")
print("\n starting task1_5")

def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
	temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
	list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

	[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

	translation_dist = [-x_min, -y_min]
	H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

	output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
	output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
	return output_img


#r1 = warpImage(img1,M)
r2= warpImages(img2,img1,M)
cv2.imwrite('task1_pano.jpg',r2)
