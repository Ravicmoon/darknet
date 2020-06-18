#pragma once

#include <opencv2/opencv.hpp>

#include "box.h"
#include "image.h"
#include "libapi.h"
#include "option_list.h"

LIB_API void Mat2Image(cv::Mat const& mat, Image* image);
LIB_API void DrawYoloDetections(cv::Mat& img, Detection* dets, int num_boxes,
    float thresh, Metadata const& md);

LIB_API cv::Mat DrawLossGraphBg(
    int max_iter, float max_loss, int num_lines, int img_size);
LIB_API void DrawLossGraph(cv::Mat const& bg, std::vector<int> const& iter,
    std::vector<float> const& avg_loss, std::vector<int> const& iter_map,
    std::vector<float> const& map, int max_iter, float max_loss,
    double time_remaining);