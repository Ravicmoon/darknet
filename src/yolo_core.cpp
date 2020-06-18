#include "yolo_core.h"

#include <stdio.h>
#include <stdlib.h>

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

DEFINE_bool(clear, false, "");
DEFINE_bool(show_imgs, false, "");
DEFINE_bool(calc_map, true, "");

DEFINE_int32(benchmark_layers, 0, "");
DEFINE_int32(letter_box, 0, "");
DEFINE_int32(map_points, 0, "");
DEFINE_int32(width, -1, "");
DEFINE_int32(height, -1, "");
DEFINE_int32(num_gpus, 1, "");
DEFINE_int32(cuda_dbg_sync, 0, "");

DEFINE_double(thresh, 0.25, "");
DEFINE_double(iou_thresh, 0.5, "");
DEFINE_double(hier_thresh, 0.5, "");

DEFINE_string(mode, "video", "");
DEFINE_string(data_file, "yolo.data", "");
DEFINE_string(model_file, "yolo.cfg", "");
DEFINE_string(weights_file, "yolo.weights", "");
DEFINE_string(input_file, "test.avi", "");

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

  if (FLAGS_mode == "train")
    TrainDetector(FLAGS_data_file.c_str(), FLAGS_model_file.c_str(),
        FLAGS_weights_file.c_str(), FLAGS_num_gpus, FLAGS_clear,
        FLAGS_show_imgs, FLAGS_calc_map, FLAGS_benchmark_layers);

  if (FLAGS_mode == "valid")
  {
    Network* net = LoadNetworkCustom(
        FLAGS_model_file.c_str(), FLAGS_weights_file.c_str(), 0, 1);

    ValidateDetector(net, FLAGS_data_file.c_str(), 0.5, 0);

    FreeNetwork(net);
    free(net);
  }

  if (FLAGS_mode == "test")
  {
  }

  if (FLAGS_mode == "video")
  {
    float const nms = 0.45f;
    int num_boxes = 0;

    Metadata md(FLAGS_data_file.c_str());
    Network* net = LoadNetworkCustom(
        FLAGS_model_file.c_str(), FLAGS_weights_file.c_str(), 0, 1);
    layer* l = &net->layers[net->n - 1];
    Image image = {0, 0, 0, nullptr};
    Detection* dets = nullptr;

    net->benchmark_layers = FLAGS_benchmark_layers;

    srand(2222222);

    cv::Mat input, resize;
    cv::VideoCapture video_capture(FLAGS_input_file);
    while (video_capture.isOpened() && video_capture.read(input))
    {
      cv::resize(input, resize, cv::Size(net->w, net->h));
      cv::cvtColor(resize, resize, cv::COLOR_RGB2BGR);
      Mat2Image(resize, &image);
      NetworkPredict(net, image.data);

      dets = GetNetworkBoxes(net, net->w, net->h, FLAGS_thresh,
          FLAGS_hier_thresh, 0, 1, &num_boxes, 0);

      if (l->nms_kind == DEFAULT_NMS)
        NmsSort(dets, num_boxes, l->classes, nms);
      else
        DiouNmsSort(dets, num_boxes, l->classes, nms, l->nms_kind, l->beta_nms);

      DrawYoloDetections(input, dets, num_boxes, FLAGS_thresh, md);

      FreeDetections(dets, num_boxes);

      cv::imshow("demo", input);
      if (cv::waitKey(1) == 27)
        break;
    }

    FreeNetwork(net);
    free(net);
  }

  return 0;
}
