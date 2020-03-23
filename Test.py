'''
'''
import glob
import math
import re
from typing import Tuple

from pprint import pprint

import cv2
import numpy as np
from matplotlib import pyplot as plt

def processImage(imageFile: str, preview: bool = 0) -> Tuple[np.ndarray, np.ndarray]:
    global pattern_size

    if not preview:
        print('#', end='')

    img = cv2.imread(imageFile)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        if preview:
            cv2.destroyAllWindows()
            for i, corner in enumerate(corners):
                x, y = corner.ravel()
                cv2.circle(img, (x, y), 10, (0, 0, i/(11*7) * 255), -1)
            cv2.imshow(imageFile, img)
            cv2.waitKey(100)

        return (pattern_points, corners)

global pattern_size

ImageFiles = glob.glob("./CalibImages/Left/*.png")[::3]

square_size = 20
pattern_size = (11, 7)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

gray = cv2.imread(ImageFiles[0], cv2.IMREAD_GRAYSCALE)
imageSize = gray.shape[::-1]

chessboards = []

threads_num = 12
if threads_num <= 1:
    chessboards = [processImage(fn, 1) for fn in ImageFiles]
else:
    print("Run with %d threads..." % threads_num)
    print('Processing Images:\n' + '#'*len(ImageFiles))
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(threads_num)
    chessboards = pool.map(processImage, ImageFiles)

chessboards = [i for i in chessboards if i is not None]

for (pattern_points, corners) in chessboards:
    objpoints.append(pattern_points)
    imgpoints.append(corners)

print('\nCalibrating ...')
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

print("\nRMS:", rms)
print("Num images: ", len(imgpoints))
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, imageSize, 0, imageSize)

# Undistort Test Image
testImage = cv2.imread('./Left_1.png')
dst = cv2.undistort(testImage, camera_matrix, dist_coefs, None, newcameramtx)
cv2.namedWindow('undist', cv2.WINDOW_NORMAL)
cv2.resizeWindow('undist', 960, 720)
cv2.imshow('undist', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
