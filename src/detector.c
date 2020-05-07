#include <stdlib.h>

#include "box.h"
#include "cost_layer.h"
#include "darknet.h"
#include "network.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

void TrainDetector(char const* data_file, char const* model_file,
    char const* weights_file, int* gpus, int ngpus, int clear, int dont_show,
    int calc_map, int show_imgs, int benchmark_layers, char* chart_path)
{
  list* options = ReadDataCfg(data_file);
  char* train_images = FindOptionStr(options, "train", "data/train.txt");
  char* valid_images = FindOptionStr(options, "valid", train_images);
  char* backup_directory = FindOptionStr(options, "backup", "/backup/");

  Network net_map;
  if (calc_map)
  {
    FILE* valid_file = fopen(valid_images, "r");
    if (!valid_file)
    {
      printf(
          "\n Error: There is no %s file for mAP calculation!\n Don't use -map "
          "flag.\n Or set valid=%s in your %s file. \n",
          valid_images, train_images, data_file);
      getchar();
      exit(-1);
    }
    else
      fclose(valid_file);

    cuda_set_device(gpus[0]);
    printf(" Prepare additional network for mAP calculation...\n");
    ParseNetworkCfgCustom(&net_map, model_file, 1, 1);
    net_map.benchmark_layers = benchmark_layers;

    for (int k = 0; k < net_map.n - 1; ++k)
    {
      free_layer_custom(net_map.layers[k], 1);
    }

    char* name_list = FindOptionStr(options, "names", "data/names.list");
    int names_size = 0;
    char** names = get_labels_custom(name_list, &names_size);

    int const num_classes = net_map.layers[net_map.n - 1].classes;
    if (num_classes != names_size)
    {
      printf(
          "\n Error: in the file %s number of names %d that isn't equal to "
          "classes=%d in the file %s \n",
          name_list, names_size, num_classes, model_file);
      if (num_classes > names_size) getchar();
    }
    free_ptrs((void**)names, net_map.layers[net_map.n - 1].classes);
  }

  srand(time(0));
  char* base = BaseCfg(model_file);
  printf("%s\n", base);
  float avg_loss = -1;
  Network* nets = (Network*)xcalloc(ngpus, sizeof(Network));

  srand(time(0));
  int seed = rand();
  for (int k = 0; k < ngpus; ++k)
  {
    srand(seed);
#ifdef GPU
    cuda_set_device(gpus[k]);
#endif
    ParseNetworkCfg(&nets[k], model_file);
    nets[k].benchmark_layers = benchmark_layers;
    if (weights_file)
    {
      LoadWeights(&nets[k], weights_file);
    }
    if (clear)
    {
      *nets[k].seen = 0;
      *nets[k].cur_iteration = 0;
    }
    nets[k].learning_rate *= ngpus;
  }
  srand(time(0));
  Network net = nets[0];

  int const actual_batch_size = net.batch * net.subdivisions;
  if (actual_batch_size == 1)
  {
    printf(
        "\n Error: You set incorrect value batch=1 for Training! You should "
        "set batch=64 subdivision=64 \n");
    getchar();
  }
  else if (actual_batch_size < 8)
  {
    printf(
        "\n Warning: You set batch=%d lower than 64! It is recommended to set "
        "batch=64 subdivision=64 \n",
        actual_batch_size);
  }

  int imgs = net.batch * net.subdivisions * ngpus;
  printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate,
      net.momentum, net.decay);
  data train, buffer;

  layer l = net.layers[net.n - 1];

  int classes = l.classes;
  float jitter = l.jitter;

  list* plist = get_paths(train_images);
  int train_images_num = plist->size;
  char** paths = (char**)list_to_array(plist);

  int const init_w = net.w;
  int const init_h = net.h;
  int const init_b = net.batch;
  int iter_save = GetCurrentIteration(&net);
  int iter_save_last = GetCurrentIteration(&net);
  int iter_map = GetCurrentIteration(&net);
  float mean_average_precision = -1;
  float best_map = mean_average_precision;

  load_args args = {0};
  args.w = net.w;
  args.h = net.h;
  args.c = net.c;
  args.paths = paths;
  args.n = imgs;
  args.m = plist->size;
  args.classes = classes;
  args.flip = net.flip;
  args.jitter = jitter;
  args.num_boxes = l.max_boxes;
  net.num_boxes = args.num_boxes;
  net.train_images_num = train_images_num;
  args.d = &buffer;
  args.type = DETECTION_DATA;
  args.threads = 64;  // 16 or 64

  args.angle = net.angle;
  args.gaussian_noise = net.gaussian_noise;
  args.blur = net.blur;
  args.mixup = net.mixup;
  args.exposure = net.exposure;
  args.saturation = net.saturation;
  args.hue = net.hue;
  args.letter_box = net.letter_box;
  if (dont_show && show_imgs) show_imgs = 2;
  args.show_imgs = show_imgs;

#ifdef OPENCV
  args.threads = 6 * ngpus;  // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8
                             // logical cores) - p3.16xlarge
  // args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
  mat_cv* img = NULL;
  float max_img_loss = 5;
  int number_of_lines = 100;
  int img_size = 1000;
  char windows_name[100];
  sprintf(windows_name, "chart_%s.png", base);
  img = draw_train_chart(windows_name, max_img_loss, net.max_batches,
      number_of_lines, img_size, dont_show, chart_path);
#endif  // OPENCV

  if (net.track)
  {
    args.track = net.track;
    args.augment_speed = net.augment_speed;
    if (net.sequential_subdivisions)
      args.threads = net.sequential_subdivisions * ngpus;
    else
      args.threads = net.subdivisions * ngpus;
    args.mini_batch = net.batch / net.time_steps;
    printf(
        "\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = "
        "%d \n",
        net.batch, net.subdivisions, net.time_steps, args.mini_batch);
  }

  pthread_t load_thread = load_data(args);

  int count = 0;
  double time_remaining, avg_time = -1, alpha_time = 0.01;

  while (GetCurrentIteration(&net) < net.max_batches)
  {
    if (l.random && count++ % 10 == 0)
    {
      float rand_coef = 1.4;
      if (l.random != 1.0) rand_coef = l.random;
      printf("Resizing, random_coef = %.2f \n", rand_coef);
      float random_val = rand_scale(rand_coef);  // *x or /x
      int dim_w =
          roundl(random_val * init_w / net.resize_step + 1) * net.resize_step;
      int dim_h =
          roundl(random_val * init_h / net.resize_step + 1) * net.resize_step;
      if (random_val < 1 && (dim_w > init_w || dim_h > init_h))
        dim_w = init_w, dim_h = init_h;

      int max_dim_w =
          roundl(rand_coef * init_w / net.resize_step + 1) * net.resize_step;
      int max_dim_h =
          roundl(rand_coef * init_h / net.resize_step + 1) * net.resize_step;

      // at the beginning (check if enough memory) and at the end (calc rolling
      // mean/variance)
      if (avg_loss < 0 || GetCurrentIteration(&net) > net.max_batches - 100)
      {
        dim_w = max_dim_w;
        dim_h = max_dim_h;
      }

      if (dim_w < net.resize_step) dim_w = net.resize_step;
      if (dim_h < net.resize_step) dim_h = net.resize_step;
      int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
      int new_dim_b = (int)(dim_b * 0.8);
      if (new_dim_b > init_b) dim_b = new_dim_b;

      args.w = dim_w;
      args.h = dim_h;

      if (net.dynamic_minibatch)
      {
        for (int k = 0; k < ngpus; ++k)
        {
          (*nets[k].seen) =
              init_b * net.subdivisions *
              GetCurrentIteration(
                  &net);  // remove this line, when you will save to
                          // weights-file both: seen & cur_iteration
          nets[k].batch = dim_b;

          for (int j = 0; j < nets[k].n; ++j)
          {
            nets[k].layers[j].batch = dim_b;
          }
        }
        net.batch = dim_b;
        imgs = net.batch * net.subdivisions * ngpus;
        args.n = imgs;
        printf("\n %d x %d  (batch = %d) \n", dim_w, dim_h, net.batch);
      }
      else
        printf("\n %d x %d \n", dim_w, dim_h);

      pthread_join(load_thread, 0);
      train = buffer;
      free_data(train);
      load_thread = load_data(args);

      for (int k = 0; k < ngpus; ++k)
      {
        resize_network(nets + k, dim_w, dim_h);
      }
      net = nets[0];
    }
    double time = what_time_is_it_now();
    pthread_join(load_thread, 0);
    train = buffer;
    if (net.track)
    {
      net.sequential_subdivisions = GetCurrentSeqSubdivisions(&net);
      args.threads = net.sequential_subdivisions * ngpus;
      printf(" sequential_subdivisions = %d, sequence = %d \n",
          net.sequential_subdivisions, GetSequenceValue(&net));
    }
    load_thread = load_data(args);

    const double load_time = (what_time_is_it_now() - time);
    printf("Loaded: %lf seconds", load_time);
    if (load_time > 0.1 && avg_loss > 0)
      printf(" - performance bottleneck on CPU or Disk HDD/SSD");
    printf("\n");

    time = what_time_is_it_now();
    float loss = 0;
#ifdef GPU
    if (ngpus == 1)
    {
      int wait_key = (dont_show) ? 0 : 1;
      loss = TrainNetworkWaitKey(&net, train, wait_key);
    }
    else
    {
      loss = TrainNetworks(nets, ngpus, train, 4);
    }
#else
    loss = TrainNetwork(net, train);
#endif
    if (avg_loss < 0 || avg_loss != avg_loss)
      avg_loss = loss;  // if(-inf or nan)
    avg_loss = avg_loss * .9 + loss * .1;

    const int iteration = GetCurrentIteration(&net);

    int calc_map_for_each =
        4 * train_images_num /
        (net.batch * net.subdivisions);  // calculate mAP for each 4 Epochs
    calc_map_for_each = fmax(calc_map_for_each, 100);
    int next_map_calc = iter_map + calc_map_for_each;
    next_map_calc = fmax(next_map_calc, net.burn_in);

    if (calc_map)
    {
      printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
      if (mean_average_precision > 0)
        printf("\n Last accuracy mAP@0.5 = %2.2f %%, best = %2.2f %% ",
            mean_average_precision * 100, best_map * 100);
    }

    if (net.cudnn_half)
    {
      if (iteration < net.burn_in * 3)
        fprintf(stderr,
            "\n Tensor Cores are disabled until the first %d iterations "
            "are reached.",
            3 * net.burn_in);
      else
        fprintf(stderr, "\n Tensor Cores are used.");
    }
    printf(
        "\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours "
        "left\n",
        iteration, loss, avg_loss, GetCurrentRate(&net),
        (what_time_is_it_now() - time), iteration * imgs, avg_time);

    int draw_precision = 0;
    if (calc_map &&
        (iteration >= next_map_calc || iteration == net.max_batches))
    {
      if (l.random)
      {
        printf("Resizing to initial size: %d x %d ", init_w, init_h);
        args.w = init_w;
        args.h = init_h;
        int k;
        if (net.dynamic_minibatch)
        {
          for (k = 0; k < ngpus; ++k)
          {
            for (k = 0; k < ngpus; ++k)
            {
              nets[k].batch = init_b;
              int j;
              for (j = 0; j < nets[k].n; ++j) nets[k].layers[j].batch = init_b;
            }
          }
          net.batch = init_b;
          imgs = init_b * net.subdivisions * ngpus;
          args.n = imgs;
          printf("\n %d x %d  (batch = %d) \n", init_w, init_h, init_b);
        }
        pthread_join(load_thread, 0);
        free_data(train);
        train = buffer;
        load_thread = load_data(args);
        for (k = 0; k < ngpus; ++k)
        {
          resize_network(nets + k, init_w, init_h);
        }
        net = nets[0];
      }

      copy_weights_net(net, &net_map);

      iter_map = iteration;
      mean_average_precision =
          ValidateDetector(data_file, model_file, weights_file, 0.25, 0.5, 0,
              net.letter_box, &net_map);  // &net_combined);
      printf("\n mean_average_precision (mAP@0.5) = %f \n",
          mean_average_precision);
      if (mean_average_precision > best_map)
      {
        best_map = mean_average_precision;
        printf("New best mAP!\n");
        char buff[256];
        sprintf(buff, "%s/%s_best.weights", backup_directory, base);
        save_weights(net, buff);
      }

      draw_precision = 1;
    }
    time_remaining = (net.max_batches - iteration) *
                     (what_time_is_it_now() - time + load_time) / 60 / 60;
    // set initial value, even if resume training from 10000 iteration
    if (avg_time < 0)
      avg_time = time_remaining;
    else
      avg_time = alpha_time * time_remaining + (1 - alpha_time) * avg_time;
#ifdef OPENCV
    draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss,
        iteration, net.max_batches, mean_average_precision, draw_precision,
        "mAP%", dont_show, avg_time);
#endif  // OPENCV

    // if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
    // if (i % 100 == 0) {
    if (iteration >= (iter_save + 1000) || iteration % 1000 == 0)
    {
      iter_save = iteration;
#ifdef GPU
      if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
      char buff[256];
      sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
      save_weights(net, buff);
    }

    if (iteration >= (iter_save_last + 100) ||
        (iteration % 100 == 0 && iteration > 1))
    {
      iter_save_last = iteration;
#ifdef GPU
      if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
      char buff[256];
      sprintf(buff, "%s/%s_last.weights", backup_directory, base);
      save_weights(net, buff);
    }
    free_data(train);
  }
#ifdef GPU
  if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
  char buff[256];
  sprintf(buff, "%s/%s_final.weights", backup_directory, base);
  save_weights(net, buff);

#ifdef OPENCV
  release_mat(&img);
  destroy_all_windows_cv();
#endif

  // free memory
  pthread_join(load_thread, 0);
  free_data(buffer);

  free_load_threads(&args);

  free(base);
  free(paths);
  free_list_contents(plist);
  free_list(plist);

  free_list_contents_kvp(options);
  free_list(options);

  for (int k = 0; k < ngpus; ++k)
  {
    FreeNetwork(&nets[k]);
  }
  free(nets);

  if (calc_map)
  {
    net_map.n = 0;
    FreeNetwork(&net_map);
  }
}

typedef struct
{
  box b;
  float p;
  int class_id;
  int image_index;
  int truth_flag;
  int unique_truth_index;
} box_prob;

int detections_comparator(const void* pa, const void* pb)
{
  box_prob a = *(const box_prob*)pa;
  box_prob b = *(const box_prob*)pb;
  float diff = a.p - b.p;
  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  return 0;
}

float ValidateDetector(char const* data_file, char const* model_file,
    char const* weights_file, float thresh_calc_avg_iou, const float iou_thresh,
    const int map_points, int letter_box, Network* existing_net)
{
  int j;
  list* options = ReadDataCfg(data_file);
  char* valid_images = FindOptionStr(options, "valid", "data/train.txt");
  char* difficult_valid_images = FindOptionStr(options, "difficult", NULL);
  char* name_list = FindOptionStr(options, "names", "data/names.list");
  int names_size = 0;
  char** names =
      get_labels_custom(name_list, &names_size);  // get_labels(name_list);
  // char *mapf = option_find_str(options, "map", 0);
  // int *map = 0;
  // if (mapf) map = read_map(mapf);
  FILE* reinforcement_fd = NULL;

  Network net;
  // int initial_batch;
  if (existing_net)
  {
    char* train_images = FindOptionStr(options, "train", "data/train.txt");
    valid_images = FindOptionStr(options, "valid", train_images);
    net = *existing_net;
    remember_network_recurrent_state(*existing_net);
    free_network_recurrent_state(*existing_net);
  }
  else
  {
    ParseNetworkCfgCustom(&net, model_file, 1, 1);  // set batch=1
    if (weights_file)
    {
      LoadWeights(&net, weights_file);
    }
    // set_batch_network(&net, 1);
    FuseConvBatchNorm(&net);
    calculate_binary_weights(net);
  }
  if (net.layers[net.n - 1].classes != names_size)
  {
    printf(
        "\n Error: in the file %s number of names %d that isn't equal to "
        "classes=%d in the file %s \n",
        name_list, names_size, net.layers[net.n - 1].classes, model_file);
    getchar();
  }
  srand(time(0));
  printf("\n calculation mAP (mean average precision)...\n");

  list* plist = get_paths(valid_images);
  char** paths = (char**)list_to_array(plist);

  char** paths_dif = NULL;
  if (difficult_valid_images)
  {
    list* plist_dif = get_paths(difficult_valid_images);
    paths_dif = (char**)list_to_array(plist_dif);
  }

  layer l = net.layers[net.n - 1];
  int classes = l.classes;

  int m = plist->size;
  int i = 0;
  int t;

  const float thresh = .005;
  const float nms = .45;
  // const float iou_thresh = 0.5;

  int nthreads = 4;
  if (m < 4) nthreads = m;
  Image* val = (Image*)xcalloc(nthreads, sizeof(Image));
  Image* val_resized = (Image*)xcalloc(nthreads, sizeof(Image));
  Image* buf = (Image*)xcalloc(nthreads, sizeof(Image));
  Image* buf_resized = (Image*)xcalloc(nthreads, sizeof(Image));
  pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

  load_args args = {0};
  args.w = net.w;
  args.h = net.h;
  args.c = net.c;
  if (letter_box)
    args.type = LETTERBOX_DATA;
  else
    args.type = IMAGE_DATA;

  // const float thresh_calc_avg_iou = 0.24;
  float avg_iou = 0;
  int tp_for_thresh = 0;
  int fp_for_thresh = 0;

  box_prob* detections = (box_prob*)xcalloc(1, sizeof(box_prob));
  int detections_count = 0;
  int unique_truth_count = 0;

  int* truth_classes_count = (int*)xcalloc(classes, sizeof(int));

  // For multi-class precision and recall computation
  float* avg_iou_per_class = (float*)xcalloc(classes, sizeof(float));
  int* tp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));
  int* fp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));

  for (t = 0; t < nthreads; ++t)
  {
    args.path = paths[i + t];
    args.im = &buf[t];
    args.resized = &buf_resized[t];
    thr[t] = load_data_in_thread(args);
  }
  time_t start = time(0);
  for (i = nthreads; i < m + nthreads; i += nthreads)
  {
    fprintf(stderr, "\r%d", i);
    for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t)
    {
      pthread_join(thr[t], 0);
      val[t] = buf[t];
      val_resized[t] = buf_resized[t];
    }
    for (t = 0; t < nthreads && (i + t) < m; ++t)
    {
      args.path = paths[i + t];
      args.im = &buf[t];
      args.resized = &buf_resized[t];
      thr[t] = load_data_in_thread(args);
    }
    for (t = 0; t < nthreads && i + t - nthreads < m; ++t)
    {
      const int image_index = i + t - nthreads;
      char* path = paths[image_index];
      char* id = BaseCfg(path);
      float* X = val_resized[t].data;
      NetworkPredict(&net, X);

      int nboxes = 0;
      float hier_thresh = 0;
      Detection* dets;
      if (args.type == LETTERBOX_DATA)
      {
        dets = GetNetworkBoxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0,
            1, &nboxes, letter_box);
      }
      else
      {
        dets = GetNetworkBoxes(
            &net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes, letter_box);
      }
      // detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh,
      // hier_thresh, 0, 1, &nboxes, letter_box); // for letter_box=1
      if (nms)
      {
        if (l.nms_kind == DEFAULT_NMS)
          do_nms_sort(dets, nboxes, l.classes, nms);
        else
          diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
      }
      // if (nms) do_nms_obj(dets, nboxes, l.classes, nms);

      char labelpath[4096];
      replace_image_to_label(path, labelpath);
      int num_labels = 0;
      box_label* truth = read_boxes(labelpath, &num_labels);
      int j;
      for (j = 0; j < num_labels; ++j)
      {
        truth_classes_count[truth[j].id]++;
      }

      // difficult
      box_label* truth_dif = NULL;
      int num_labels_dif = 0;
      if (paths_dif)
      {
        char* path_dif = paths_dif[image_index];

        char labelpath_dif[4096];
        replace_image_to_label(path_dif, labelpath_dif);

        truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
      }

      const int checkpoint_detections_count = detections_count;

      int i;
      for (i = 0; i < nboxes; ++i)
      {
        int class_id;
        for (class_id = 0; class_id < classes; ++class_id)
        {
          float prob = dets[i].prob[class_id];
          if (prob > 0)
          {
            detections_count++;
            detections = (box_prob*)xrealloc(
                detections, detections_count * sizeof(box_prob));
            detections[detections_count - 1].b = dets[i].bbox;
            detections[detections_count - 1].p = prob;
            detections[detections_count - 1].image_index = image_index;
            detections[detections_count - 1].class_id = class_id;
            detections[detections_count - 1].truth_flag = 0;
            detections[detections_count - 1].unique_truth_index = -1;

            int truth_index = -1;
            float max_iou = 0;
            for (j = 0; j < num_labels; ++j)
            {
              box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
              // printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d
              // \n",
              //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
              float current_iou = box_iou(dets[i].bbox, t);
              if (current_iou > iou_thresh && class_id == truth[j].id)
              {
                if (current_iou > max_iou)
                {
                  max_iou = current_iou;
                  truth_index = unique_truth_count + j;
                }
              }
            }

            // best IoU
            if (truth_index > -1)
            {
              detections[detections_count - 1].truth_flag = 1;
              detections[detections_count - 1].unique_truth_index = truth_index;
            }
            else
            {
              // if object is difficult then remove detection
              for (j = 0; j < num_labels_dif; ++j)
              {
                box t = {truth_dif[j].x, truth_dif[j].y, truth_dif[j].w,
                    truth_dif[j].h};
                float current_iou = box_iou(dets[i].bbox, t);
                if (current_iou > iou_thresh && class_id == truth_dif[j].id)
                {
                  --detections_count;
                  break;
                }
              }
            }

            // calc avg IoU, true-positives, false-positives for required
            // Threshold
            if (prob > thresh_calc_avg_iou)
            {
              int z, found = 0;
              for (z = checkpoint_detections_count; z < detections_count - 1;
                   ++z)
              {
                if (detections[z].unique_truth_index == truth_index)
                {
                  found = 1;
                  break;
                }
              }

              if (truth_index > -1 && found == 0)
              {
                avg_iou += max_iou;
                ++tp_for_thresh;
                avg_iou_per_class[class_id] += max_iou;
                tp_for_thresh_per_class[class_id]++;
              }
              else
              {
                fp_for_thresh++;
                fp_for_thresh_per_class[class_id]++;
              }
            }
          }
        }
      }

      unique_truth_count += num_labels;

      // static int previous_errors = 0;
      // int total_errors = fp_for_thresh + (unique_truth_count -
      // tp_for_thresh); int errors_in_this_image = total_errors -
      // previous_errors; previous_errors = total_errors; if(reinforcement_fd ==
      // NULL) reinforcement_fd = fopen("reinforcement.txt", "wb"); char
      // buff[1000]; sprintf(buff, "%s\n", path); if(errors_in_this_image > 0)
      // fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

      free_detections(dets, nboxes);
      free(id);
      free_image(val[t]);
      free_image(val_resized[t]);
    }
  }

  // for (t = 0; t < nthreads; ++t) {
  //    pthread_join(thr[t], 0);
  //}

  if ((tp_for_thresh + fp_for_thresh) > 0)
    avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

  int class_id;
  for (class_id = 0; class_id < classes; class_id++)
  {
    if ((tp_for_thresh_per_class[class_id] +
            fp_for_thresh_per_class[class_id]) > 0)
      avg_iou_per_class[class_id] =
          avg_iou_per_class[class_id] / (tp_for_thresh_per_class[class_id] +
                                            fp_for_thresh_per_class[class_id]);
  }

  // SORT(detections)
  qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

  typedef struct
  {
    double precision;
    double recall;
    int tp, fp, fn;
  } pr_t;

  // for PR-curve
  pr_t** pr = (pr_t**)xcalloc(classes, sizeof(pr_t*));
  for (i = 0; i < classes; ++i)
  {
    pr[i] = (pr_t*)xcalloc(detections_count, sizeof(pr_t));
  }
  printf("\n detections_count = %d, unique_truth_count = %d  \n",
      detections_count, unique_truth_count);

  int* detection_per_class_count = (int*)xcalloc(classes, sizeof(int));
  for (j = 0; j < detections_count; ++j)
  {
    detection_per_class_count[detections[j].class_id]++;
  }

  int* truth_flags = (int*)xcalloc(unique_truth_count, sizeof(int));

  int rank;
  for (rank = 0; rank < detections_count; ++rank)
  {
    if (rank % 100 == 0)
      printf(" rank = %d of ranks = %d \r", rank, detections_count);

    if (rank > 0)
    {
      int class_id;
      for (class_id = 0; class_id < classes; ++class_id)
      {
        pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
        pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
      }
    }

    box_prob d = detections[rank];
    // if (detected && isn't detected before)
    if (d.truth_flag == 1)
    {
      if (truth_flags[d.unique_truth_index] == 0)
      {
        truth_flags[d.unique_truth_index] = 1;
        pr[d.class_id][rank].tp++;  // true-positive
      }
      else
        pr[d.class_id][rank].fp++;
    }
    else
    {
      pr[d.class_id][rank].fp++;  // false-positive
    }

    for (i = 0; i < classes; ++i)
    {
      const int tp = pr[i][rank].tp;
      const int fp = pr[i][rank].fp;
      const int fn = truth_classes_count[i] -
                     tp;  // false-negative = objects - true-positive
      pr[i][rank].fn = fn;

      if ((tp + fp) > 0)
        pr[i][rank].precision = (double)tp / (double)(tp + fp);
      else
        pr[i][rank].precision = 0;

      if ((tp + fn) > 0)
        pr[i][rank].recall = (double)tp / (double)(tp + fn);
      else
        pr[i][rank].recall = 0;

      if (rank == (detections_count - 1) &&
          detection_per_class_count[i] != (tp + fp))
      {  // check for last rank
        printf(
            " class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n",
            i, detection_per_class_count[i], tp + fp, tp, fp);
      }
    }
  }

  free(truth_flags);

  double mean_average_precision = 0;

  for (i = 0; i < classes; ++i)
  {
    double avg_precision = 0;

    // MS COCO - uses 101-Recall-points on PR-chart.
    // PascalVOC2007 - uses 11-Recall-points on PR-chart.
    // PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
    // ImageNet - uses Area-Under-Curve on PR-chart.

    // correct mAP calculation: ImageNet, PascalVOC 2010-2012
    if (map_points == 0)
    {
      double last_recall = pr[i][detections_count - 1].recall;
      double last_precision = pr[i][detections_count - 1].precision;
      for (rank = detections_count - 2; rank >= 0; --rank)
      {
        double delta_recall = last_recall - pr[i][rank].recall;
        last_recall = pr[i][rank].recall;

        if (pr[i][rank].precision > last_precision)
        {
          last_precision = pr[i][rank].precision;
        }

        avg_precision += delta_recall * last_precision;
      }
    }
    // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
    else
    {
      int point;
      for (point = 0; point < map_points; ++point)
      {
        double cur_recall = point * 1.0 / (map_points - 1);
        double cur_precision = 0;
        for (rank = 0; rank < detections_count; ++rank)
        {
          if (pr[i][rank].recall >= cur_recall)
          {  // > or >=
            if (pr[i][rank].precision > cur_precision)
            {
              cur_precision = pr[i][rank].precision;
            }
          }
        }
        // printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision =
        // %.4f \n", i, point, cur_recall, cur_precision);

        avg_precision += cur_precision;
      }
      avg_precision = avg_precision / map_points;
    }

    printf("class_id = %d, name = %s, ap = %2.2f%%   \t (TP = %d, FP = %d) \n",
        i, names[i], avg_precision * 100, tp_for_thresh_per_class[i],
        fp_for_thresh_per_class[i]);

    float class_precision =
        (float)tp_for_thresh_per_class[i] /
        ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
    float class_recall =
        (float)tp_for_thresh_per_class[i] /
        ((float)tp_for_thresh_per_class[i] +
            (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
    // printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n",
    // class_precision, class_recall, avg_iou_per_class[i]);

    mean_average_precision += avg_precision;
  }

  const float cur_precision =
      (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
  const float cur_recall =
      (float)tp_for_thresh /
      ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
  const float f1_score =
      2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
  printf(
      "\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score "
      "= %1.2f \n",
      thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

  printf(
      " for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = "
      "%2.2f %% \n",
      thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh,
      unique_truth_count - tp_for_thresh, avg_iou * 100);

  mean_average_precision = mean_average_precision / classes;
  printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
  if (map_points)
    printf("used %d Recall-points \n", map_points);
  else
    printf("used Area-Under-Curve for each unique Recall \n");

  printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh,
      mean_average_precision, mean_average_precision * 100);

  for (i = 0; i < classes; ++i)
  {
    free(pr[i]);
  }
  free(pr);
  free(detections);
  free(truth_classes_count);
  free(detection_per_class_count);

  free(avg_iou_per_class);
  free(tp_for_thresh_per_class);
  free(fp_for_thresh_per_class);

  fprintf(stderr, "Total Detection Time: %d Seconds\n", (int)(time(0) - start));
  printf("\nSet -points flag:\n");
  printf(" `-points 101` for MS COCO \n");
  printf(
      " `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) "
      "\n");
  printf(
      " `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom "
      "dataset\n");
  if (reinforcement_fd != NULL) fclose(reinforcement_fd);

  // free memory
  free_ptrs((void**)names, net.layers[net.n - 1].classes);
  free_list_contents_kvp(options);
  free_list(options);

  if (existing_net)
  {
    // set_batch_network(&net, initial_batch);
    // free_network_recurrent_state(*existing_net);
    restore_network_recurrent_state(*existing_net);
    // randomize_network_recurrent_state(*existing_net);
  }
  else
  {
    FreeNetwork(&net);
  }
  if (val) free(val);
  if (val_resized) free(val_resized);
  if (thr) free(thr);
  if (buf) free(buf);
  if (buf_resized) free(buf_resized);

  return mean_average_precision;
}

void TestDetector(char const* data_file, char const* model_file,
    char const* weights_file, char const* filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels,
    char* outfile, int letter_box, int benchmark_layers)
{
  list* options = ReadDataCfg(data_file);
  char* name_list = FindOptionStr(options, "names", "data/names.list");
  int names_size = 0;
  char** names = get_labels_custom(name_list, &names_size);

  Network net = {0};
  ParseNetworkCfgCustom(&net, model_file, 1, 1);
  if (weights_file)
  {
    LoadWeights(&net, weights_file);
  }
  net.benchmark_layers = benchmark_layers;
  FuseConvBatchNorm(&net);
  calculate_binary_weights(net);
  if (net.layers[net.n - 1].classes != names_size)
  {
    printf(
        "\n Error: in the file %s number of names %d that isn't equal to "
        "classes=%d in the file %s \n",
        name_list, names_size, net.layers[net.n - 1].classes, model_file);
    if (net.layers[net.n - 1].classes > names_size) getchar();
  }
  srand(2222222);
  char buff[256];
  char* input = buff;
  char* json_buf = NULL;
  int json_image_id = 0;
  FILE* json_file = NULL;
  if (outfile)
  {
    json_file = fopen(outfile, "wb");
    if (!json_file)
    {
      error("fopen failed");
    }
    char* tmp = "[\n";
    fwrite(tmp, sizeof(char), strlen(tmp), json_file);
  }
  int j;
  float nms = .45;  // 0.4F
  while (1)
  {
    if (filename)
    {
      strncpy(input, filename, 256);
      if (strlen(input) > 0)
        if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
    }
    else
    {
      printf("Enter Image Path: ");
      fflush(stdout);
      input = fgets(input, 256, stdin);
      if (!input) break;
      strtok(input, "\n");
    }
    // image im;
    // image sized = load_image_resize(input, net.w, net.h, net.c, &im);
    Image im = load_image(input, 0, 0, net.c);
    Image sized;
    if (letter_box)
      sized = letterbox_image(im, net.w, net.h);
    else
      sized = resize_image(im, net.w, net.h);
    layer l = net.layers[net.n - 1];

    // box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    // float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
    // for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes,
    // sizeof(float));

    float* X = sized.data;

    // time= what_time_is_it_now();
    double time = get_time_point();
    NetworkPredict(&net, X);
    // network_predict_image(&net, im); letterbox = 1;
    printf("%s: Predicted in %lf milli-seconds.\n", input,
        ((double)get_time_point() - time) / 1000);
    // printf("%s: Predicted in %f seconds.\n", input,
    // (what_time_is_it_now()-time));

    int nboxes = 0;
    Detection* dets = GetNetworkBoxes(
        &net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
    if (nms)
    {
      if (l.nms_kind == DEFAULT_NMS)
        do_nms_sort(dets, nboxes, l.classes, nms);
      else
        diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
    }
    draw_detections_v3(
        im, dets, nboxes, thresh, names, NULL, l.classes, ext_output);
    save_image(im, "predictions");
    if (!dont_show)
    {
      show_image(im, "predictions");
    }

    if (json_file)
    {
      if (json_buf)
      {
        char* tmp = ", \n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
      }
      ++json_image_id;
      json_buf = detection_to_json(
          dets, nboxes, l.classes, names, json_image_id, input);

      fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
      free(json_buf);
    }

    // pseudo labeling concept - fast.ai
    if (save_labels)
    {
      char labelpath[4096];
      replace_image_to_label(input, labelpath);

      FILE* fw = fopen(labelpath, "wb");
      int i;
      for (i = 0; i < nboxes; ++i)
      {
        char buff[1024];
        int class_id = -1;
        float prob = 0;
        for (j = 0; j < l.classes; ++j)
        {
          if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob)
          {
            prob = dets[i].prob[j];
            class_id = j;
          }
        }
        if (class_id >= 0)
        {
          sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id,
              dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
          fwrite(buff, sizeof(char), strlen(buff), fw);
        }
      }
      fclose(fw);
    }

    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);

    if (!dont_show)
    {
      wait_until_press_key_cv();
      destroy_all_windows_cv();
    }

    if (filename) break;
  }

  if (json_file)
  {
    char* tmp = "\n]";
    fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    fclose(json_file);
  }

  // free memory
  free_ptrs((void**)names, net.layers[net.n - 1].classes);
  free_list_contents_kvp(options);
  free_list(options);

  FreeNetwork(&net);
}
