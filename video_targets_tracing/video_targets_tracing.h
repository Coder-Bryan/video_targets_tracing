#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <vector>

// TODO: General description of what this method do
std::vector<cv::Rect2d> trackWithModel(cv::dnn::Net& net, cv::Mat& frame, std::vector<cv::String> targetname,
	const float confidence_threshold, const float nms_threshold, 
	std::vector<cv::Scalar> color);
