#include "yolo_core.h"

#include <stdio.h>
#include <stdlib.h>
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include <algorithm>

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
  size_t step = mat.step;

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

float GetRandColor(int c, int x, int max)
{
  static float const colors[6][3] = {
      {1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};

  float ratio = ((float)x / max) * 5.0f;
  int i = floor(ratio);
  int j = ceil(ratio);
  ratio -= i;

  return (1 - ratio) * colors[i][c] + ratio * colors[j][c];
}

cv::Scalar GetRandColor(int offset, int max)
{
  float r = 256 * GetRandColor(2, offset, max);
  float g = 256 * GetRandColor(1, offset, max);
  float b = 256 * GetRandColor(0, offset, max);

  return cv::Scalar(int(r), int(g), int(b));
}

void DrawYoloDetections(cv::Mat& img, Detection* dets, int num_boxes,
    float thresh, Metadata const& md)
{
  for (int i = 0; i < num_boxes; i++)
  {
    std::string label;
    int class_id = -1;
    for (int j = 0; j < md.NumClasses(); j++)
    {
      if (dets[i].prob[j] < thresh)
        continue;

      if (class_id < 0)
      {
        class_id = j;
        label += md.NameAt(j);

        char prob[10];
        sprintf(prob, "(%2.0f%%)", dets[i].prob[j] * 100);
        label += prob;
      }
      else
      {
        label += ", " + md.NameAt(j);
      }
    }

    if (class_id >= 0)
    {
      Box::AbsBox b(dets[i].bbox);
      float left = b.left * img.cols;
      float right = b.right * img.cols;
      float top = b.top * img.rows;
      float bottom = b.bottom * img.rows;

      int font_face = cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL;
      cv::Size text_size = cv::getTextSize(label, font_face, 1, 1, 0);

      cv::Point2f pt1(left, top);
      cv::Point2f pt2(right, bottom);
      cv::Point2f pt_text(left, top - 4);
      cv::Point2f pt_text_bg1(left, top - 21);
      cv::Point2f pt_text_bg2(right, top);
      if (right - left < text_size.width)
        pt_text_bg2.x = left + text_size.width;

      int offset = class_id * 123457 % md.NumClasses();
      cv::Scalar color = GetRandColor(offset, md.NumClasses());

      int width = (int)std::max(1.0f, img.rows * 0.002f);

      cv::rectangle(img, pt1, pt2, color, width, 8, 0);
      cv::rectangle(img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
      cv::rectangle(img, pt_text_bg1, pt_text_bg2, color, -1, 8, 0);
      cv::putText(
          img, label, pt_text, font_face, 1, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
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
