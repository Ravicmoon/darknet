#include "darknet.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "dark_cuda.h"
#include "data.h"
#include "demo.h"
#include "option_list.h"
#include "utils.h"

int main(int argc, char** argv)
{
#ifdef _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

  for (int i = 0; i < argc; ++i)
  {
    if (!argv[i]) continue;
    strip_args(argv[i]);
  }

  if (argc < 4)
  {
    fprintf(stderr,
        "usage: %s [train/test/valid/demo/map] [data] [cfg] [weights "
        "(optional)]\n",
        argv[0]);
    return -1;
  }

  gpu_index = find_int_arg(argc, argv, "-i", 0);
  if (find_arg(argc, argv, "-nogpu"))
  {
    gpu_index = -1;
    printf(
        "\n Currently Darknet doesn't support -nogpu flag. If you want to use "
        "CPU - please compile Darknet with GPU=0 in the Makefile, or compile "
        "darknet_no_gpu.sln on Windows.\n");
    return -2;
  }

#ifndef GPU
  gpu_index = -1;
  printf(" GPU isn't used \n");
  init_cpu();
#else  // GPU
  if (gpu_index >= 0)
  {
    cuda_set_device(gpu_index);
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  }

  show_cuda_cudnn_info();
  cuda_debug_sync = find_arg(argc, argv, "-cuda_debug_sync");

#ifdef CUDNN_HALF
  printf(" CUDNN_HALF=1 \n");
#endif  // CUDNN_HALF
#endif  // GPU

  int dont_show = find_arg(argc, argv, "-dont_show");
  int benchmark = find_arg(argc, argv, "-benchmark");
  int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
  if (benchmark) dont_show = 1;
  int letter_box = find_arg(argc, argv, "-letter_box");
  int calc_map = find_arg(argc, argv, "-map");
  int map_points = find_int_arg(argc, argv, "-points", 0);
  int show_imgs = find_arg(argc, argv, "-show_imgs");
  int width = find_int_arg(argc, argv, "-width", -1);
  int height = find_int_arg(argc, argv, "-height", -1);
  int ext_output = find_arg(argc, argv, "-ext_output");
  int save_labels = find_arg(argc, argv, "-save_labels");
  int clear = find_arg(argc, argv, "-clear");
  int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
  int frame_skip = find_int_arg(argc, argv, "-frame_skip", 0);
  int cam_index = find_int_arg(argc, argv, "-cam_index", 0);

  float thresh = find_float_arg(argc, argv, "-thresh", .25);
  float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);
  float hier_thresh = find_float_arg(argc, argv, "-hier", .5);

  char* out_filename = find_char_arg(argc, argv, "-out_filename", 0);
  char* outfile = find_char_arg(argc, argv, "-out", 0);
  char* chart_path = find_char_arg(argc, argv, "-chart", 0);
  char* gpu_list = find_char_arg(argc, argv, "-gpus", 0);
  char* prefix = find_char_arg(argc, argv, "-prefix", 0);

  int* gpus = 0;
  int gpu = 0;
  int ngpus = 0;
  if (gpu_list)
  {
    printf("%s\n", gpu_list);
    int len = (int)strlen(gpu_list);
    ngpus = 1;
    int i;
    for (i = 0; i < len; ++i)
    {
      if (gpu_list[i] == ',') ++ngpus;
    }
    gpus = (int*)xcalloc(ngpus, sizeof(int));
    for (i = 0; i < ngpus; ++i)
    {
      gpus[i] = atoi(gpu_list);
      gpu_list = strchr(gpu_list, ',') + 1;
    }
  }
  else
  {
    gpu = gpu_index;
    gpus = &gpu;
    ngpus = 1;
  }

  char* datacfg = argv[2];
  char* cfg = argv[3];
  char* weights = (argc > 4) ? argv[4] : 0;
  if (weights)
    if (strlen(weights) > 0)
      if (weights[strlen(weights) - 1] == 0x0d)
        weights[strlen(weights) - 1] = 0;
  char* filename = (argc > 5) ? argv[5] : 0;

  if (0 == strcmp(argv[1], "train"))
    train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show,
        calc_map, show_imgs, benchmark_layers, chart_path);
  else if (0 == strcmp(argv[1], "test"))
    test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh,
        dont_show, ext_output, save_labels, outfile, letter_box,
        benchmark_layers);
  else if (0 == strcmp(argv[1], "map"))
    validate_detector_map(datacfg, cfg, weights, thresh, iou_thresh, map_points,
        letter_box, NULL);
  else if (0 == strcmp(argv[1], "demo"))
  {
    list* options = read_data_cfg(datacfg);
    int classes = option_find_int(options, "classes", 20);
    char* name_list = option_find_str(options, "names", "data/names.list");
    char** names = get_labels(name_list);
    if (filename)
      if (strlen(filename) > 0)
        if (filename[strlen(filename) - 1] == 0x0d)
          filename[strlen(filename) - 1] = 0;

    demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes,
        frame_skip, prefix, out_filename, dont_show, ext_output, letter_box,
        time_limit_sec, benchmark, benchmark_layers);

    free_list_contents_kvp(options);
    free_list(options);
  }
  else
    printf(" There isn't such command: %s", argv[1]);

  if (gpus && gpu_list && ngpus > 1) free(gpus);

  return 0;
}
