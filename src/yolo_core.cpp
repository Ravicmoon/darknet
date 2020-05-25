#include "yolo_core.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#ifdef GPU
#include <cuda.h>
#endif

#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

DEFINE_int32(dont_show, 0, "");
DEFINE_int32(benchmark, 0, "");
DEFINE_int32(benchmark_layers, 0, "");
DEFINE_int32(letter_box, 0, "");
DEFINE_int32(calc_map, 0, "");
DEFINE_int32(map_points, 0, "");
DEFINE_int32(show_imgs, 0, "");
DEFINE_int32(width, -1, "");
DEFINE_int32(height, -1, "");
DEFINE_int32(clear, 0, "");
DEFINE_int32(gpu_idx, 0, "");
DEFINE_int32(cuda_dbg_sync, 0, "");

DEFINE_double(thresh, 0.25, "");
DEFINE_double(iou_thresh, 0.5, "");
DEFINE_double(hier_thresh, 0.5, "");

DEFINE_string(mode, "video", "");
DEFINE_string(data_file, "", "");
DEFINE_string(model_file, "", "");
DEFINE_string(weights_file, "", "");
DEFINE_string(chart_path, "", "");
DEFINE_string(input_file, "", "");
DEFINE_string(gpu_list, "", "");

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

void Mat2Image(cv::Mat const& mat, Image* image)
{
  int w = mat.cols;
  int h = mat.rows;
  int c = mat.channels();

  if (image->data == nullptr)
  {
    image->w = w;
    image->h = h;
    image->c = c;
    image->data = new float[h * w * c];
  }

  unsigned char* data = (unsigned char*)mat.data;
  int step = mat.step;

  for (int y = 0; y < h; y++)
  {
    for (int k = 0; k < c; k++)
    {
      for (int x = 0; x < w; x++)
      {
        image->data[k * w * h + y * w + x] =
            data[y * step + x * c + k] / 255.0f;
      }
    }
  }
}

int main(int argc, char** argv)
{
#ifdef _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);

#ifndef GPU
  FLAGS_gpu_idx = -1;
  printf("GPU isn't used\n");
  init_cpu();
#else   // GPU
  if (FLAGS_gpu_idx >= 0)
  {
    cudaSetDevice(FLAGS_gpu_idx);
    CUDA_ASSERT(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  }
  ShowCudaCudnnInfo();
#endif  // GPU

  int gpu = FLAGS_gpu_idx;
  int* gpus = &gpu;
  int ngpus = 1;

  if (FLAGS_mode == "train")
    TrainDetector(FLAGS_data_file.c_str(), FLAGS_model_file.c_str(),
        FLAGS_weights_file.c_str(), FLAGS_chart_path.c_str(), gpus, ngpus,
        FLAGS_clear, FLAGS_show_imgs, FLAGS_dont_show, FLAGS_calc_map,
        FLAGS_benchmark_layers);

  if (FLAGS_mode == "val")
    ValidateDetector(FLAGS_data_file.c_str(), FLAGS_model_file.c_str(),
        FLAGS_weights_file.c_str(), FLAGS_thresh, FLAGS_iou_thresh,
        FLAGS_map_points, FLAGS_letter_box, nullptr);

  if (FLAGS_mode == "test")
  {
  }

  if (FLAGS_mode == "video")
  {
    Metadata md = GetMetadata(FLAGS_data_file.c_str());

    Network* net = LoadNetworkCustom(
        FLAGS_model_file.c_str(), FLAGS_weights_file.c_str(), 0, 1);
    layer* l = &net->layers[net->n - 1];
    net->benchmark_layers = FLAGS_benchmark_layers;
    srand(2222222);

    float const nms = 0.45f;
    Image image = {0, 0, 0, nullptr};
    Detection* detection = nullptr;
    int num_boxes = 0;

    cv::Mat input, resize;
    cv::VideoCapture video_capture(FLAGS_input_file);
    while (video_capture.isOpened() && video_capture.read(input))
    {
      cv::resize(input, resize, cv::Size(net->w, net->h));
      cv::cvtColor(resize, resize, cv::COLOR_RGB2BGR);
      Mat2Image(resize, &image);
      NetworkPredict(net, image.data);

      detection = GetNetworkBoxes(net, net->w, net->h, FLAGS_thresh,
          FLAGS_hier_thresh, 0, 1, &num_boxes, 0);

      if (l->nms_kind == DEFAULT_NMS)
        do_nms_sort(detection, num_boxes, l->classes, nms);
      else
        diounms_sort(
            detection, num_boxes, l->classes, nms, l->nms_kind, l->beta_nms);

      for (int i = 0; i < num_boxes; i++)
      {
        for (int j = 0; j < l->classes; j++)
        {
          if (detection[i].prob[j] < FLAGS_thresh)
            continue;

          box b = detection[i].bbox;
          float left = (b.x - b.w / 2.0f) * input.cols;
          float right = (b.x + b.w / 2.0f) * input.cols;
          float top = (b.y - b.h / 2.0f) * input.rows;
          float bottom = (b.y + b.h / 2.0f) * input.rows;

          cv::rectangle(input, cv::Point2f(left, top),
              cv::Point2f(right, bottom), CV_RGB(255, 0, 0), 2);
        }
      }

      FreeDetections(detection, num_boxes);

      cv::imshow("demo", input);
      if (cv::waitKey(1) == 27)
        break;
    }

    // free_ptrs((void**)md.names, md.classes);

    FreeNetwork(net);
    free(net);
  }

  return 0;
}
