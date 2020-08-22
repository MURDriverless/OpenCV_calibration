#pragma once

#include <opencv2/core.hpp>
#include <vector>

void calibratePoints(cv::Size boardSize, double squareSize, cv::Size imageSize, std::vector<std::vector<cv::Point2f>> foundPoints);
void calcBoardCornerPos(cv::Size boardSize, double squareSize, std::vector<cv::Point3f>& corners);