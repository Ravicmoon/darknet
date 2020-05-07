#include "darknet.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

#include "dark_cuda.h"
#include "data.h"
#include "option_list.h"
#include "utils.h"

DEFINE_int32(dont_show, 0, "");
DEFINE_int32(benchmark, 0, "");
DEFINE_int32(benchmark_layers, 0, "");
DEFINE_int32(letter_box, 0, "");
DEFINE_int32(calc_map, 0, "");
DEFINE_int32(map_points, 0, "");
DEFINE_int32(show_imgs, 0, "");
DEFINE_int32(width, -1, "");
DEFINE_int32(height, -1, "");
DEFINE_int32(ext_output, 0, "");
DEFINE_int32(save_labels, 0, "");
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
DEFINE_string(filename, "", "");
DEFINE_string(outfile, "", "");
DEFINE_string(out_filename, "", "");
DEFINE_string(gpu_list, "", "");

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
  printf(" GPU isn't used \n");
  init_cpu();
#else  // GPU
  if (FLAGS_gpu_idx >= 0)
  {
    cuda_set_device(FLAGS_gpu_idx);
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  }

  show_cuda_cudnn_info();
  cuda_debug_sync = FLAGS_cuda_dbg_sync;

#ifdef CUDNN_HALF
  printf(" CUDNN_HALF=1 \n");
#endif  // CUDNN_HALF
#endif  // GPU

  int gpu = FLAGS_gpu_idx;
  int* gpus = &gpu;
  int ngpus = 1;

  char* chart_path = strdup(FLAGS_chart_path.c_str());
  char* filename = strdup(FLAGS_filename.c_str());
  char* outfile = strdup(FLAGS_outfile.c_str());
  char* out_filename = strdup(FLAGS_out_filename.c_str());

  if (FLAGS_mode == "train")
    TrainDetector(FLAGS_data_file.c_str(), FLAGS_model_file.c_str(),
        FLAGS_weights_file.c_str(), gpus, ngpus, FLAGS_clear, FLAGS_dont_show,
        FLAGS_calc_map, FLAGS_show_imgs, FLAGS_benchmark_layers, chart_path);

  if (FLAGS_mode == "val")
    ValidateDetector(FLAGS_data_file.c_str(), FLAGS_model_file.c_str(),
        FLAGS_weights_file.c_str(), FLAGS_thresh, FLAGS_iou_thresh,
        FLAGS_map_points, FLAGS_letter_box, nullptr);

  if (FLAGS_mode == "test")
    TestDetector(FLAGS_data_file.c_str(), FLAGS_model_file.c_str(),
        FLAGS_weights_file.c_str(), FLAGS_filename.c_str(), FLAGS_thresh,
        FLAGS_hier_thresh, FLAGS_dont_show, FLAGS_ext_output, FLAGS_save_labels,
        outfile, FLAGS_letter_box, FLAGS_benchmark_layers);

  if (FLAGS_mode == "video")
  {
    Metadata md = GetMetadata(FLAGS_data_file.c_str());

    Network* net = LoadNetworkCustom(
        FLAGS_model_file.c_str(), FLAGS_weights_file.c_str(), FLAGS_clear, 1);
    layer l = net->layers[net->n - 1];
    net->benchmark_layers = FLAGS_benchmark_layers;
    srand(2222222);

    float const nms = 0.45f;
    Image image = {0, 0, 0, nullptr};
    Detection* detection = nullptr;
    int num_boxes = 0;

    cv::Mat input, resize;
    cv::VideoCapture video_capture(FLAGS_filename);
    while (video_capture.isOpened() && video_capture.read(input))
    {
      cv::resize(input, resize, cv::Size(net->w, net->h));
      cv::cvtColor(resize, resize, cv::COLOR_RGB2BGR);
      Mat2Image(resize, &image);
      NetworkPredict(net, image.data);

      detection = GetNetworkBoxes(net, net->w, net->h, FLAGS_thresh,
          FLAGS_hier_thresh, 0, 1, &num_boxes, 0);

      if (l.nms_kind == DEFAULT_NMS)
        do_nms_sort(detection, num_boxes, l.classes, nms);
      else
        diounms_sort(
            detection, num_boxes, l.classes, nms, l.nms_kind, l.beta_nms);

      for (int i = 0; i < num_boxes; i++)
      {
        for (int j = 0; j < l.classes; j++)
        {
          if (detection[i].prob[j] < FLAGS_thresh) continue;

          box b = detection[i].bbox;
          float left = (b.x - b.w / 2.0f) * input.cols;
          float right = (b.x + b.w / 2.0f) * input.cols;
          float top = (b.y - b.h / 2.0f) * input.rows;
          float bottom = (b.y + b.h / 2.0f) * input.rows;

          cv::rectangle(input, cv::Point2f(left, top),
              cv::Point2f(right, bottom), CV_RGB(255, 0, 0), 2);
        }
      }

      free_detections(detection, num_boxes);

      cv::imshow("demo", input);
      if (cv::waitKey(1) == 27) break;
    }

    FreeNetwork(net);
    free(net);
  }

  free(out_filename);
  free(outfile);
  free(filename);
  free(chart_path);

  return 0;
}
