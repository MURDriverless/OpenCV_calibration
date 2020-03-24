import os

import cv2
import numpy as np

def UndistortImage(CalibrationPath: str, ImagePath: str, OutputPath: str = '', ImageAlpha: float = 1):
    with np.load(CalibrationPath) as calibFile:
        camera_matrix = calibFile['camera_matrix']
        dist_coefs = calibFile['dist_coefs']

        if os.path.isfile(ImagePath):
            imageName = os.path.split(ImagePath)[-1]
        else:
            raise FileNotFoundError('ImagePath:{} is not a file'.format(ImagePath))

        image = cv2.imread(ImagePath)
        imageSize = image.shape[1::-1]

        if not OutputPath:
            OutputPath = os.path.dirname(ImagePath)
            OutputPath = os.path.join(OutputPath, 'output')
        elif os.path.isdir(OutputPath):
            raise NotADirectoryError('OutputPath:{} is not a directory'.format(OutputPath))

        os.makedirs(OutputPath, exist_ok = True)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, imageSize, ImageAlpha, imageSize)
        newImage = cv2.undistort(image, camera_matrix, dist_coefs, None, newcameramtx)
        cv2.imwrite(os.path.join(OutputPath, imageName), newImage)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process image with calibrated undistortion.')
    parser.add_argument('CalibrationPath', help='Path to calibration file, .npz')
    parser.add_argument('ImagePath', help='Path to image file, .npz')
    parser.add_argument('--OutputPath', '-o', help='Path to output folder', default='')
    parser.add_argument('--ImageAlpha', '-a', help='OpenCV undistortion alpha (0 to 1)', default = 1)
    args = vars(parser.parse_args())

    UndistortImage(**args)