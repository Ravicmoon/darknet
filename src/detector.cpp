#include <float.h>
#include <stdlib.h>

#include "box.h"
#include "cost_layer.h"
#include "image.h"
#include "image_opencv.h"
#include "network.h"
#include "option_list.h"
#include "parser.h"
#include "utils.h"
#include "visualize.h"
#include "yolo_core.h"

void SaveWeights(Network* net, std::string save_dir, std::string save_filename,
    std::string postfix)
{
  std::stringstream ss;
  ss << save_dir << '/' << save_filename;
  if (!postfix.empty())
    ss << '_' << postfix;
  ss << ".weights";

  SaveWeights(net, ss.str().c_str());
}

void TrainDetector(Metadata const& md, std::string model_file,
    std::string weights_file, int num_gpus, bool clear, bool show_imgs,
    bool calc_map, int benchmark_layers)
{
  std::string save_dir = md.SaveDir();
  if (!Exists(save_dir.c_str()))
    MakeDir(save_dir.c_str(), 0);

  // make a gpu index array
  std::vector<int> gpus;
  for (int i = 0; i < num_gpus; i++)
  {
    gpus.push_back(i);
  }

  Network* net_map = nullptr;
  if (calc_map)
  {
#ifdef GPU
    cuda_set_device(gpus[0]);
#endif
    net_map = (Network*)calloc(1, sizeof(Network));
    LoadNetwork(net_map, model_file.c_str(), nullptr, false, true);
    for (int i = 0; i < net_map->n; i++)
    {
      free_layer(&net_map->layers[i], true);
    }
  }

  srand(time(0));

  Network* nets = (Network*)calloc(num_gpus, sizeof(Network));
  for (int k = 0; k < num_gpus; ++k)
  {
#ifdef GPU
    cuda_set_device(gpus[k]);
#endif
    LoadNetwork(
        nets + k, model_file.c_str(), weights_file.c_str(), true, clear);
    nets[k].benchmark_layers = benchmark_layers;
    nets[k].lr *= num_gpus;
  }

  Network* net = &nets[0];
  printf("Learning rate: %e, Momentum: %g, Decay: %g\n", net->lr, net->momentum,
      net->decay);

  list* train_img_paths = get_paths(md.TrainFile().c_str());
  char** paths = (char**)ListToArray(train_img_paths);
  int const num_train_imgs = train_img_paths->size;

  int const actual_batch = net->batch * net->subdiv;
  int const img_per_step = actual_batch * num_gpus;
  int const iter_per_epoch = (int)((float)num_train_imgs / actual_batch + 0.5f);

  assert(actual_batch > 8);

  // set max number of iterations
  net->max_iter = iter_per_epoch * net->max_epoch;
  printf("Max number of iterations: %d\n", net->max_iter);

  int const init_w = net->w;
  int const init_h = net->h;
  int iter_save = GetCurrIter(net);
  int iter_map = max_val_cmp(net->burn_in, GetCurrIter(net));

  layer* l = &net->layers[net->n - 1];
  data train, buffer;

  load_args args = {0};
  args.w = net->w;
  args.h = net->h;
  args.c = net->c;
  args.paths = paths;
  args.n = img_per_step;
  args.m = num_train_imgs;
  args.classes = l->classes;
  args.flip = net->flip;
  args.jitter = l->jitter;
  args.num_boxes = l->max_boxes;
  args.d = &buffer;
  args.type = DETECTION_DATA;

  args.gaussian_noise = net->gaussian_noise;
  args.blur = net->blur;
  args.mixup = net->mixup;
  args.exposure = net->exposure;
  args.saturation = net->saturation;
  args.hue = net->hue;
  args.show_imgs = show_imgs;

  args.threads = 6 * num_gpus;

  pthread_t load_thread = load_data(args);

  int const max_loss = 5;
  cv::Mat graph_bg = DrawLossGraphBg(net->max_iter, max_loss, 100, 720);

  int count = 0;
  double avg_time = -DBL_MAX;
  float avg_loss = -FLT_MAX;
  float best_map = 0.0f;

  double const alpha_time = 0.01;
  int const map_step = max_val_cmp(100, iter_per_epoch);

  int last_idx = model_file.find_last_of(".");
  std::string save_filename = model_file.substr(0, last_idx);

  std::vector<float> avg_loss_stack, map_stack;
  std::vector<int> iter_stack, iter_map_stack;

  while (GetCurrIter(net) < net->max_iter)
  {
    if (l->random && count++ % 10 == 0)
    {
      float rand_coef = 1.4f;
      if (abs(l->random - 1.0f) > FLT_EPSILON)
        rand_coef = l->random;

      float scale = RandScale(rand_coef);
      int dim_w =
          roundl(scale * init_w / net->resize_step + 1) * net->resize_step;
      int dim_h =
          roundl(scale * init_h / net->resize_step + 1) * net->resize_step;

      int max_dim_w =
          roundl(rand_coef * init_w / net->resize_step + 1) * net->resize_step;
      int max_dim_h =
          roundl(rand_coef * init_h / net->resize_step + 1) * net->resize_step;

      // at the beginning (check if enough memory)
      // at the end (calc rolling mean/variance)
      if (avg_loss < 0 || GetCurrIter(net) > net->max_iter - 100)
      {
        dim_w = max_dim_w;
        dim_h = max_dim_h;
      }

      dim_w = std::max(dim_w, net->resize_step);
      dim_h = std::max(dim_h, net->resize_step);

      args.w = dim_w;
      args.h = dim_h;

      printf("Resizing: %d x %d\n", dim_w, dim_h);

      pthread_join(load_thread, nullptr);
      train = buffer;
      free_data(train);
      load_thread = load_data(args);

      for (int k = 0; k < num_gpus; ++k)
      {
        ResizeNetwork(nets + k, dim_w, dim_h);
      }
    }

    double start_step = GetTimePoint();
    pthread_join(load_thread, nullptr);

    train = buffer;
    load_thread = load_data(args);

    float loss = 0;
#ifdef GPU
    if (num_gpus == 1)
      loss = TrainNetwork(net, train);
    else
      loss = TrainNetworks(nets, num_gpus, train, 4);
#else
    loss = TrainNetwork(net, train);
#endif

    if (avg_loss < 0)
      avg_loss = loss;
    avg_loss = avg_loss * 0.9f + loss * 0.1f;
    avg_loss_stack.push_back(avg_loss);

    int const iter = GetCurrIter(net);
    iter_stack.push_back(iter);

    if (net->cudnn_half)
    {
      if (iter < net->burn_in * 3)
        printf("Tensor Cores are disabled until the first %d iterations\n",
            3 * net->burn_in);
      else
        printf("Tensor Cores are used\n");
    }

    if (calc_map && (iter >= iter_map || iter == net->max_iter))
    {
      if (l->random)
      {
        printf("Resizing to initial size: %d x %d ", init_w, init_h);
        args.w = init_w;
        args.h = init_h;

        pthread_join(load_thread, nullptr);
        free_data(train);
        train = buffer;
        load_thread = load_data(args);
        for (int k = 0; k < num_gpus; ++k)
        {
          ResizeNetwork(nets + k, init_w, init_h);
        }
      }

      CopyNetWeights(net, net_map);

      float map = ValidateDetector(md, net_map, 0.5);

      map_stack.push_back(map);
      iter_map_stack.push_back(iter_map);

      if (map > best_map)
      {
        best_map = map;
        SaveWeights(net, save_dir.c_str(), save_filename, "best");
      }

      iter_map = iter + map_step;

      std::cout << "Next mAP calculation: " << iter_map << std::endl;
      std::cout << "Best mAP = " << best_map << std::endl;
    }

    double end_step = GetTimePoint();
    double time_remaining = ((net->max_iter - iter) / num_gpus) *
                            (end_step - start_step) / 1e6 / 3600;

    if (avg_time < 0)
      avg_time = time_remaining;
    else
      avg_time = alpha_time * time_remaining + (1 - alpha_time) * avg_time;

    printf(
        "[%04d] loss: %.2f, avg loss: %.2f, lr: %e, images: %d, %.2lf hours "
        "left\n",
        iter, loss, avg_loss, GetCurrLr(net), net->seen, avg_time);

    DrawLossGraph(graph_bg, iter_stack, avg_loss_stack, iter_map_stack,
        map_stack, net->max_iter, max_loss, avg_time);

    if (iter >= (iter_save + 1000) || iter % 1000 == 0)
    {
      iter_save = iter;
#ifdef GPU
      if (num_gpus != 1)
        SyncNetworks(nets, num_gpus, 0);
#endif
      SaveWeights(net, save_dir.c_str(), save_filename, std::to_string(iter));
    }

    free_data(train);
  }

#ifdef GPU
  if (num_gpus != 1)
    SyncNetworks(nets, num_gpus, 0);
#endif
  SaveWeights(net, save_dir.c_str(), save_filename, "final");

  cv::destroyAllWindows();

  // free memory
  pthread_join(load_thread, nullptr);
  free_data(buffer);

  free_load_threads(&args);

  free(paths);
  FreeListContents(train_img_paths);
  FreeList(train_img_paths);

  for (int k = 0; k < num_gpus; ++k)
  {
    FreeNetwork(&nets[k]);
  }
  free(nets);

  if (net_map != nullptr)
  {
    net_map->n = 0;
    FreeNetwork(net_map);
    free(net_map);
  }
}

typedef struct
{
  Box b;
  float p;
  int cid;
  int gt_idx;
  bool matched;
} ValBox;

float ValidateDetector(Metadata const& md, Network* net, float const iou_thresh)
{
  std::vector<std::string> name_list = md.NameList();

  layer* l = &net->layers[net->n - 1];
  int classes = l->classes;
  if (classes != name_list.size())
  {
    printf(" # of names %d is not equal to  # of classes %d\n",
        (int)name_list.size(), classes);
    return -1.0f;
  }

  float const thresh = .005;
  float const nms = .45;

  Image* buff = new Image;
  Image* buff_resized = new Image;

  load_args args = {0};
  args.w = net->w;
  args.h = net->h;
  args.c = net->c;
  args.type = IMAGE_DATA;
  args.im = buff;
  args.resized = buff_resized;

  std::vector<ValBox> val_boxes;
  std::vector<int> num_gt_class(classes, 0);
  std::vector<int> num_pred_class(classes, 0);
  int num_gt = 0;

  std::vector<std::string> val_img_list = md.ValImgList();

  double start = GetTimePoint();
  for (size_t i = 0; i < val_img_list.size(); i++)
  {
    printf("\rCalculating mAP for %d samples...", i);

    args.path = val_img_list[i].c_str();
    pthread_t thr = load_data_in_thread(args);
    pthread_join(thr, nullptr);

    NetworkPredict(net, buff_resized->data);

    free_image(*buff);
    free_image(*buff_resized);

    int num_boxes = 0;
    Detection* dets = GetNetworkBoxes(net, 1, 1, thresh, 0, &num_boxes);

    if (l->nms_kind == DEFAULT_NMS)
      NmsSort(dets, num_boxes, l->classes, nms);
    else
      DiouNmsSort(dets, num_boxes, l->classes, nms, l->nms_kind, l->beta_nms);

    std::string label_path = ReplaceImage2Label(val_img_list[i]);
    std::vector<BoxLabel> gt = ReadBoxAnnot(label_path);
    for (size_t k = 0; k < gt.size(); ++k)
    {
      num_gt_class[gt[k].id]++;
    }

    for (int j = 0; j < num_boxes; ++j)
    {
      for (int cid = 0; cid < classes; ++cid)
      {
        Box pred_box = dets[j].bbox;
        float pred_prob = dets[j].prob[cid];
        if (abs(pred_prob) < FLT_EPSILON)
          continue;

        num_pred_class[cid]++;

        int gt_idx = -1;
        float max_iou = 0;
        for (size_t k = 0; k < gt.size(); ++k)
        {
          Box gt_box(gt[k].x, gt[k].y, gt[k].w, gt[k].h);
          float iou = Box::Iou(pred_box, gt_box);
          if (iou > iou_thresh && iou > max_iou && cid == gt[k].id)
          {
            max_iou = iou;
            gt_idx = num_gt + k;
          }
        }

        ValBox v;
        v.b = pred_box;
        v.p = pred_prob;
        v.cid = cid;
        if (gt_idx > -1)
          v.matched = true;
        else
          v.matched = false;
        v.gt_idx = gt_idx;

        val_boxes.push_back(v);
      }
    }

    num_gt += (int)gt.size();

    FreeDetections(dets, num_boxes);
  }

  delete buff, buff_resized;

  printf("\n # of pred: %d, # of GT: %d\n", val_boxes.size(), num_gt);

  // calculating precision-recall curve
  std::sort(val_boxes.begin(), val_boxes.end(),
      [&](ValBox const& v1, ValBox const& v2) -> bool {
        if (v1.p > v2.p)
          return true;
        else
          return false;
      });

  typedef struct
  {
    double precision;
    double recall;
    int tp, fp, fn;
  } pr_t;

  std::vector<std::vector<pr_t>> pr(classes);
  for (size_t i = 0; i < pr.size(); i++)
  {
    pr[i].resize(val_boxes.size());
  }

  std::vector<bool> gt_flags(num_gt, false);
  for (size_t i = 0; i < val_boxes.size(); ++i)
  {
    if (i > 0)
    {
      for (int cid = 0; cid < classes; ++cid)
      {
        pr[cid][i].tp = pr[cid][i - 1].tp;
        pr[cid][i].fp = pr[cid][i - 1].fp;
      }
    }

    ValBox v = val_boxes[i];
    if (v.matched)
    {
      if (!gt_flags[v.gt_idx])
      {
        gt_flags[v.gt_idx] = true;
        pr[v.cid][i].tp++;
      }
      else
      {
        pr[v.cid][i].fp++;
      }
    }
    else
    {
      pr[v.cid][i].fp++;
    }

    for (int cid = 0; cid < classes; ++cid)
    {
      int const tp = pr[cid][i].tp;
      int const fp = pr[cid][i].fp;
      int const fn = num_gt_class[cid] - tp;
      pr[cid][i].fn = fn;

      if (tp + fp > 0)
        pr[cid][i].precision = (double)tp / (tp + fp);
      else
        pr[cid][i].precision = 0;

      if (tp + fn > 0)
        pr[cid][i].recall = (double)tp / (tp + fn);
      else
        pr[cid][i].recall = 0;

      if (i == val_boxes.size() - 1 && num_pred_class[cid] != tp + fp)
      {
        printf("# of predictions is not matched with tp + fp: %d != %d",
            num_pred_class[cid], tp + fp);
        return -1.0f;
      }
    }
  }

  double map = 0.0;
  for (int cid = 0; cid < classes; ++cid)
  {
    double ap = 0;
    double last_recall = pr[cid].back().recall;
    double last_precision = pr[cid].back().precision;
    for (auto it = pr[cid].rbegin(); it != pr[cid].rend(); it++)
    {
      double delta_recall = last_recall - it->recall;
      last_recall = it->recall;
      last_precision = std::max(last_precision, it->precision);

      ap += delta_recall * last_precision;
    }

    printf(
        " cid = %d, name = %s, ap = %2.2f%%\n", cid, name_list[cid], ap * 100);

    map += ap;
  }
  map = map / classes;

  printf(" mAP@%0.2f = %f, or %2.2f%%\n", iou_thresh, map, map * 100);
  printf(" Spent time: %.2lf s\n", (GetTimePoint() - start) / 1e6);

  return map;
}