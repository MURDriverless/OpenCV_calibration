#include "cvCalib.hpp"

#include <iostream>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

void calibratePoints(cv::Size boardSize, double squareSize, cv::Size imageSize, std::vector<std::vector<cv::Point2f>> foundPoints) {
    std::vector<cv::Mat> rvecs, tvecs;

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    cv::Mat stdDeviationsIntrinsics;
    cv::Mat stdDeviationsExtrinsics;
    cv::Mat perViewErrors;

    std::string outputPath = "./calibration.xml";

    std::vector<std::vector<cv::Point3f>> objPoints(1);
    calcBoardCornerPos(boardSize, squareSize, objPoints[0]);
    objPoints.resize(foundPoints.size(), objPoints[0]);

    double reprojErr = cv::calibrateCamera(objPoints, foundPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors);

    std::cout << "---" << std::endl;
    std::cout << "Calibration done, RMS: " << reprojErr << std::endl;

    std::ostringstream commentString;

    time_t now;
    std::time(&now);
    commentString << "Calibration date: " << std::asctime(localtime(&now)) << std::endl;
    commentString << "Number of frames: " << foundPoints.size() << std::endl;
    commentString << "RMS: " << reprojErr << std::endl;

    cv::FileStorage fs;
    fs.open("./calibration.xml", cv::FileStorage::WRITE);
    fs.writeComment(commentString.str());
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distCoeffs" << distCoeffs;
    fs << "calibImageSize" << imageSize;
    fs.release();

    std::cout << "Saved to " << outputPath << std::endl;
}

void calcBoardCornerPos(cv::Size boardSize, double squareSize, std::vector<cv::Point3f>& corners) {
    corners.clear();

    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            corners.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));
        }
    }
}
