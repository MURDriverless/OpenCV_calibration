#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "cvCalib.hpp"
#include "ezGlob.hpp"

#ifdef USE_OMP
#include <omp.h>
#else
#define omp_get_num_threads() 1
#endif // ifdef USE_OMP

int main(int argc, char** argv) {
    std::vector<std::string> imagePaths = glob("../calibImages/*.png");

    double squareSize = 20;
    cv::Size boardSize(11, 7);
    cv::Size imageSize;

    const double pixelSize = 3.45E-6; // [m]
    const double focalLength = 5E-3;  // [m]

    std::vector<std::vector<cv::Point2f>> foundPoints;

    #pragma omp parallel for
    for (int i = 0; i < imagePaths.size(); i++) {
        if (i == 0) {
            std::cout << "Threads = " << omp_get_num_threads() << std::endl;
        }

        const std::string &imagePath = imagePaths[i];
        cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
        imageSize = image.size();

        std::vector<cv::Point2f> chessboardPoints;

        bool foundBoard = cv::findChessboardCorners(image, boardSize, chessboardPoints, cv::CALIB_CB_FAST_CHECK);
        if (foundBoard) {
            cv::drawChessboardCorners(image, boardSize, chessboardPoints, foundBoard);
            cv::cornerSubPix(image, chessboardPoints, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::Type::EPS | cv::TermCriteria::Type::MAX_ITER, 30, 0.1 ));

            #pragma omp critical
            foundPoints.push_back(chessboardPoints);
        }
    }

    std::cout << "---" << std::endl;
    std::cout << "Calibration Details" << std::endl;
    std::cout << "Image Size: " << imageSize << std::endl;
    std::cout << "Num. Calib Images: " << foundPoints.size() << std::endl;
    std::cout << "---" << std::endl;
    std::cout << "Calibrating" << std::endl;

    CalibArgs calibArgs;
    calibArgs.cameraMatrix = cv::Mat::zeros(cv::Size2d(3, 3), CV_64FC1);
    calibArgs.cameraMatrix.at<double>(0,0) = focalLength/pixelSize;
    calibArgs.cameraMatrix.at<double>(1,1) = focalLength/pixelSize;
    calibArgs.cameraMatrix.at<double>(2,2) = 1;
    calibArgs.cameraMatrix.at<double>(0,2) = imageSize.width/2;
    calibArgs.cameraMatrix.at<double>(1,2) = imageSize.height/2;

    calibArgs.flags |= cv::CALIB_USE_INTRINSIC_GUESS;
    calibArgs.flags |= cv::CALIB_ZERO_TANGENT_DIST;

    calibArgs.isDefault = false;

    calibratePoints(boardSize, squareSize, imageSize, foundPoints, calibArgs);

    return 0;
}