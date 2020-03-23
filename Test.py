'''
'''
import glob
import math
import re

from pprint import pprint

import cv2
import numpy as np
from matplotlib import pyplot as plt

def processImage(imageFile: str):
    img = cv2.imread(imageFile)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (11, 7), None)

    if ret:
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 10, (0, 0, i/(11*7) * 255), -1)

        return (pattern_points, corners)

ImageFiles = glob.glob("./CalibImages/Left/*.png")

square_size = 20
pattern_size = (11, 7)
pattern_points = np.zeros((11*7, 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points[:, 0] -= 5
pattern_points[:, 1] = pattern_size[1] - pattern_points[:, 1]
pattern_points *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

gray = cv2.imread(ImageFiles[0], cv2.IMREAD_GRAYSCALE)
imageSize = gray.shape[::-1]

chessboards = []

threads_num = 4
if threads_num <= 1:
    chessboards = [processImage(fn) for fn in ImageFiles]
else:
    print("Run with %d threads..." % threads_num)
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(threads_num)
    chessboards = pool.map(processImage, ImageFiles)

for (pattern_points, corners) in chessboards:
    objpoints.append(pattern_points)
    imgpoints.append(corners)

rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

print("\nRMS:", rms)
print("Num images: ", len(imgpoints))
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, imageSize, 1, imageSize)

# undistort
testImage = cv2.imread('./Left_0.png')
dst = cv2.undistort(testImage, camera_matrix, dist_coefs, None, newcameramtx)
cv2.namedWindow('undist', cv2.WINDOW_NORMAL)
cv2.resizeWindow('undist', 960, 720)
cv2.imshow('undist', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
