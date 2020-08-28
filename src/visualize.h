#pragma once

#include <opencv2/opencv.hpp>

#include "box.h"
#include "image.h"
#include "libapi.h"
#include "option_list.h"
#include "track_manager.h"

LIB_API void Mat2Image(cv::Mat const& mat, Image* image);
LIB_API void DrawYoloDetections(
    cv::Mat& img, std::vector<MostProbDet> const& dets, Metadata const& md);
LIB_API void DrawYoloTrackings(
    cv::Mat& img, std::vector<yc::Track> const& tracks, Metadata const& md);
LIB_API void DrawProcTime(cv::Mat& img, int64_t millisec);
LIB_API void DrawFrameInfo(cv::Mat& img, int64_t curr_frame, int64_t max_frame);

LIB_API cv::Mat DrawLossGraphBg(
    int max_iter, float max_loss, int num_lines, int img_size);
LIB_API void DrawLossGraph(cv::Mat const& bg, std::vector<int> const& iter,
    std::vector<float> const& avg_loss, std::vector<int> const& iter_map,
    std::vector<float> const& map, int max_iter, float max_loss,
    double time_remaining);