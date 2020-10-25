#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct CalibArgs {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    int flags = 0;
    bool isDefault = true;
};

void calibratePoints(cv::Size boardSize, double squareSize, cv::Size imageSize, std::vector<std::vector<cv::Point2f>> foundPoints, const CalibArgs& calibArgs = CalibArgs());
void calcBoardCornerPos(cv::Size boardSize, double squareSize, std::vector<cv::Point3f>& corners);