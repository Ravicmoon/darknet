#include "yolo_core.h"

#include <stdio.h>
#include <stdlib.h>

#include "geo_info.h"
#include "track_manager.h"
#include "visualize.h"

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include <algorithm>

#ifdef GPU
#include <cuda.h>
#endif

#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

DEFINE_bool(clear, false, "Clear weights in model");
DEFINE_bool(show_imgs, false, "");
DEFINE_bool(save_output, false, "Save output to image or video");
DEFINE_bool(calc_map, true, "Calculate mAP during training");
DEFINE_bool(disable_tracking, false, "Disable tracking while processing video");

DEFINE_int32(benchmark_layers, 0, "Indexes of layers to be benchmarked");
DEFINE_int32(num_gpus, 1, "Number of GPUs");
DEFINE_int32(cuda_dbg_sync, 0, "");

DEFINE_double(thresh, 0.5, "Threshold for object's confidence");
DEFINE_double(nms_thresh, 0.45, "Threshold for non-maxima suppression");

DEFINE_string(mode, "video", "Either train/valid/image/video");
DEFINE_string(data_file, "yolo.data", "Data file path");
DEFINE_string(model_file, "yolo.cfg", "Model file path");
DEFINE_string(weights_file, "yolo.weights", "Weights file path");
DEFINE_string(input_file, "test.avi",
    "Input file path for image/video modes; use comma to input multiple files");

#ifdef GPU
#define CUDA_ASSERT(x) CudaAssert((x), __FILE__, __LINE__)

void CudaAssert(cudaError_t code, char const* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA assert: %s %s %d\n", cudaGetErrorString(code), file,
        line);
    if (abort)
      exit(code);
  }
}

void ShowCudaCudnnInfo()
{
  int cuda_version = 0;
  int cuda_driver_version = 0;
  int device_count = 0;

  CUDA_ASSERT(cudaRuntimeGetVersion(&cuda_version));
  CUDA_ASSERT(cudaDriverGetVersion(&cuda_driver_version));

  fprintf(stderr, "CUDA-version: %d (%d)", cuda_version, cuda_driver_version);
  if (cuda_version > cuda_driver_version)
    fprintf(stderr, "\nWarning: CUDA-version is higher than driver-version!\n");

#ifdef CUDNN
  fprintf(
      stderr, ", cuDNN: %d.%d.%d", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
#endif  // CUDNN

#ifdef CUDNN_HALF
  fprintf(stderr, ", CUDNN_HALF=1");
#endif  // CUDNN_HALF

  CUDA_ASSERT(cudaGetDeviceCount(&device_count));
  fprintf(stderr, ", GPU count: %d\n", device_count);
}
#endif

void SeparateInputFiles(std::vector<std::string>& files)
{
  int offset = 0;
  while (1)
  {
    int idx = FLAGS_input_file.find(',', offset);
    if (idx == std::string::npos)
    {
      std::string substr = FLAGS_input_file.substr(offset);
      if (!substr.empty())
        files.push_back(substr);
      break;
    }

    files.push_back(FLAGS_input_file.substr(offset, idx - offset));
    offset = idx + 1;
  }
}

void ProcImage(Metadata const& md, Network* net, cv::Mat const& input,
    cv::Mat& resize, cv::Mat& display, Image& image,
    yc::TrackManager* track_manager = nullptr)
{
  cv::resize(input, resize, cv::Size(net->w, net->h));
  cv::resize(input, display, display.size());
  cv::cvtColor(resize, resize, cv::COLOR_RGB2BGR);
  Mat2Image(resize, &image);
  NetworkPredict(net, image.data);

  int num_dets = 0;
  Detection* dets = GetNetworkBoxes(net, FLAGS_thresh, &num_dets);

  layer* l = &net->layers[net->n - 1];
  NmsSort(
      dets, num_dets, l->classes, FLAGS_nms_thresh, l->nms_kind, l->beta_nms);

  std::vector<MostProbDet> most_prob_dets = GetMostProbDets(dets, num_dets);

  if (track_manager != nullptr)
  {
    std::vector<yc::Track*> tracks;
    track_manager->Track(most_prob_dets);
    track_manager->GetTracks(tracks);

    DrawYoloTrackings(display, tracks, md);
  }
  else
  {
    DrawYoloDetections(display, most_prob_dets, md);
  }

  FreeDetections(dets, num_dets);
}

int main(int argc, char** argv)
{
#ifdef _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);

#ifndef GPU
  printf("GPU isn't used\n");
  init_cpu();
#else   // GPU
  ShowCudaCudnnInfo();
#endif  // GPU

  Metadata md(FLAGS_data_file);

  if (FLAGS_mode == "train")
  {
    TrainDetector(md, FLAGS_model_file.c_str(), FLAGS_weights_file.c_str(),
        FLAGS_num_gpus, FLAGS_clear, FLAGS_show_imgs, FLAGS_calc_map,
        FLAGS_benchmark_layers);
  }
  else
  {
    Network* net = (Network*)calloc(1, sizeof(Network));
    LoadNetwork(net, FLAGS_model_file.c_str(), FLAGS_weights_file.c_str());

    cv::Mat resize, display;
    Image image = {0, 0, 0, nullptr};

    // calculate mAP@0.5
    if (FLAGS_mode == "valid")
    {
      ValidateDetector(md, net, 0.5);
      return 0;
    }

    std::vector<std::string> files;
    SeparateInputFiles(files);

    std::vector<yc::GeoInfo> geo_infos(files.size());
    for (size_t i = 0; i < files.size(); i++)
    {
      std::string const& path = files[i];
      std::cout << path << std::endl;

      int start = path.find_last_of('_');
      int end = path.find_last_of('.');

      std::string xml_path = path.substr(start + 1, path.size() - end) + "xml";
      geo_infos[i].Load(xml_path);
    }

    // processing a single image
    if (FLAGS_mode == "image")
    {
      if (files.size() > 1)
        FLAGS_input_file = files.front();

      cv::Mat input = cv::imread(FLAGS_input_file);
      display = cv::Mat::zeros(input.size(), CV_8UC3);

      using namespace std::chrono;
      auto start = system_clock::now();
      ///
      ProcImage(md, net, input, resize, display, image);
      ///
      auto end = system_clock::now();

      DrawProcTime(display, duration_cast<milliseconds>(end - start).count());

      cv::imshow(FLAGS_mode, display);
      cv::waitKey(0);
    }

    if (files.size() > 1)
      FLAGS_mode = "multi-video";

    // processing video stream
    if (FLAGS_mode == "video")
    {
      net->benchmark_layers = FLAGS_benchmark_layers;

      cv::VideoCapture video_capture(FLAGS_input_file);
      cv::VideoWriter writer;

      int img_width = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
      int img_height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
      double fps = video_capture.get(cv::CAP_PROP_FPS);

      display = cv::Mat::zeros(img_height / 2, img_width / 2, CV_8UC3);

      if (FLAGS_save_output)
      {
        int idx = FLAGS_input_file.find_last_of('.');
        std::string output_file = FLAGS_input_file.substr(0, idx) + "_out.mp4";

        writer.open(output_file, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
            fps, display.size());
      }

      int64_t curr_frame = 0;
      int64_t max_frame = video_capture.get(cv::CAP_PROP_FRAME_COUNT);

      int min_conf = (int)(fps / 5);
      yc::ConfParam conf_param(1, min_conf, 2 * min_conf);
      yc::TrackManager track_manager(conf_param, fps, 0.3);

      cv::Mat input;
      while (video_capture.isOpened() && video_capture.read(input))
      {
        using namespace std::chrono;
        auto start = system_clock::now();
        ///
        if (FLAGS_disable_tracking)
          ProcImage(md, net, input, resize, display, image);
        else
          ProcImage(md, net, input, resize, display, image, &track_manager);
        ///
        auto end = system_clock::now();

        if (FLAGS_save_output)
          writer << display;

        DrawProcTime(display, duration_cast<milliseconds>(end - start).count());
        DrawFrameInfo(display, curr_frame++, max_frame);

        cv::imshow(FLAGS_mode, display);
        if (cv::waitKey(1) == 27)
          break;
      }

      if (image.data != nullptr)
        delete[] image.data;
    }

    if (FLAGS_mode == "multi-video")
    {
      std::vector<cv::VideoCapture> video_captures(files.size());
      for (size_t i = 0; i < files.size(); i++)
      {
        video_captures[i].open(files[i]);
      }

      int img_width = video_captures.front().get(cv::CAP_PROP_FRAME_WIDTH);
      int img_height = video_captures.front().get(cv::CAP_PROP_FRAME_HEIGHT);
      double fps = video_captures.front().get(cv::CAP_PROP_FPS);

      display = cv::Mat::zeros(img_height / files.size(), img_width, CV_8UC3);

      cv::VideoWriter writer;
      if (FLAGS_save_output)
      {
        int idx = files.front().find_last_of('.');
        std::string output_file = files.front().substr(0, idx) + "_out.mp4";

        writer.open(output_file, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
            fps, display.size());
      }

      int64_t curr_frame = 0;
      int64_t max_frame = video_captures.front().get(cv::CAP_PROP_FRAME_COUNT);

      int min_conf = (int)(fps / 5);
      yc::ConfParam conf_param(1, min_conf, 2 * min_conf);

      Image* images = new Image[files.size()];
      std::vector<yc::TrackManager*> track_managers(files.size());
      std::vector<cv::Mat> inputs(files.size());
      std::vector<cv::Mat> resizes(files.size());
      std::vector<cv::Mat> displays(files.size());
      for (size_t i = 0; i < track_managers.size(); i++)
      {
        images[i].data = nullptr;
        displays[i] = cv::Mat::zeros(
            img_height / files.size(), img_width / files.size(), CV_8UC3);
        track_managers[i] = new yc::TrackManager(conf_param, fps, 0.3);
      }

      bool run = true;
      while (1)
      {
        for (size_t i = 0; i < video_captures.size(); i++)
        {
          if (video_captures[i].isOpened())
            video_captures[i].read(inputs[i]);
          else
            run = false;
        }

        if (!run)
          break;

        // if (curr_frame++ % files.size() != 0)
        //   continue;
        curr_frame++;

        using namespace std::chrono;
        auto start = system_clock::now();
        ///
        for (size_t i = 0; i < inputs.size(); i++)
        {
          if (FLAGS_disable_tracking)
            ProcImage(md, net, inputs[i], resizes[i], displays[i], images[i]);
          else
            ProcImage(md, net, inputs[i], resizes[i], displays[i], images[i],
                track_managers[i]);

          std::vector<yc::Track*> tracks;
          track_managers[i]->GetTracks(tracks);

          geo_infos[i].Proc(tracks);
          geo_infos[i].Draw(displays[i]);
        }
        yc::Handover* h1 = geo_infos[1].GetHandoverRegion(0);
        yc::Handover* h2 = geo_infos[0].GetHandoverRegion(1);
        yc::Handover::Crosstalk(h1, h2);
        ///
        auto end = system_clock::now();

        for (size_t i = 0; i < displays.size(); i++)
        {
          for (int y = 0; y < displays[i].rows; y++)
          {
            for (int x = 0; x < displays[i].cols; x++)
            {
              display.at<cv::Vec3b>(y, x + displays[i].cols * i) =
                  displays[i].at<cv::Vec3b>(y, x);
            }
          }
        }

        if (FLAGS_save_output)
          writer << display;

        DrawProcTime(display, duration_cast<milliseconds>(end - start).count());
        DrawFrameInfo(display, curr_frame, max_frame);

        cv::imshow(FLAGS_mode, display);
        if (cv::waitKey(1) == 27)
          break;
      }

      for (size_t i = 0; i < files.size(); i++)
      {
        if (images[i].data != nullptr)
          delete[] images[i].data;
      }
      delete[] images;

      if (image.data != nullptr)
        delete[] image.data;
    }

    FreeNetwork(net);
    free(net);
  }

  return 0;
}
