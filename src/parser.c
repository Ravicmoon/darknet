#include "parser.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation_layer.h"
#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gaussian_yolo_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "option_list.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "reorg_old_layer.h"
#include "route_layer.h"
#include "scale_channels_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "upsample_layer.h"
#include "utils.h"
#include "version.h"
#include "yolo_layer.h"

typedef struct Section
{
  char* type;
  list* options;
} Section;

void FreeSection(Section* s)
{
  free(s->type);
  node* n = s->options->front;
  while (n)
  {
    kvp* pair = (kvp*)n->val;
    free(pair->key);
    free(pair);
    node* next = n->next;
    free(n);
    n = next;
  }
  free(s->options);
  free(s);
}

list* ReadSections(char const* filename)
{
  FILE* file = fopen(filename, "r");
  if (file == NULL)
    FileError(filename);

  char* line;
  int line_num = 0;

  list* sections = MakeList();
  Section* current = 0;
  while ((line = fgetl(file)) != 0)
  {
    ++line_num;
    strip(line);
    switch (line[0])
    {
      case '[':
        current = (Section*)xmalloc(sizeof(Section));
        InsertList(sections, current);
        current->options = MakeList();
        current->type = line;
        break;
      case '\0':
      case '#':
      case ';':
        free(line);
        break;
      default:
        if (!ReadOption(line, current->options))
        {
          fprintf(stderr, "Config file error line %d, could parse: %s\n",
              line_num, line);
          free(line);
        }
        break;
    }
  }
  fclose(file);

  return sections;
}

LAYER_TYPE StrToLayerType(char* type)
{
  if (strcmp(type, "[shortcut]") == 0)
    return SHORTCUT;
  if (strcmp(type, "[scale_channels]") == 0)
    return SCALE_CHANNELS;
  if (strcmp(type, "[crop]") == 0)
    return CROP;
  if (strcmp(type, "[cost]") == 0)
    return COST;
  if (strcmp(type, "[detection]") == 0)
    return DETECTION;
  if (strcmp(type, "[region]") == 0)
    return REGION;
  if (strcmp(type, "[yolo]") == 0)
    return YOLO;
  if (strcmp(type, "[Gaussian_yolo]") == 0)
    return GAUSSIAN_YOLO;
  if (strcmp(type, "[local]") == 0)
    return LOCAL;
  if (strcmp(type, "[conv]") == 0 || strcmp(type, "[convolutional]") == 0)
    return CONVOLUTIONAL;
  if (strcmp(type, "[activation]") == 0)
    return ACTIVE;
  if (strcmp(type, "[net]") == 0 || strcmp(type, "[network]") == 0)
    return NETWORK;
  if (strcmp(type, "[conn]") == 0 || strcmp(type, "[connected]") == 0)
    return CONNECTED;
  if (strcmp(type, "[max]") == 0 || strcmp(type, "[maxpool]") == 0)
    return MAXPOOL;
  if (strcmp(type, "[reorg3d]") == 0)
    return REORG;
  if (strcmp(type, "[reorg]") == 0)
    return REORG_OLD;
  if (strcmp(type, "[avg]") == 0 || strcmp(type, "[avgpool]") == 0)
    return AVGPOOL;
  if (strcmp(type, "[dropout]") == 0)
    return DROPOUT;
  if (strcmp(type, "[batchnorm]") == 0)
    return BATCHNORM;
  if (strcmp(type, "[soft]") == 0 || strcmp(type, "[softmax]") == 0)
    return SOFTMAX;
  if (strcmp(type, "[route]") == 0)
    return ROUTE;
  if (strcmp(type, "[upsample]") == 0)
    return UPSAMPLE;
  if (strcmp(type, "[empty]") == 0)
    return EMPTY;
  return BLANK;
}

typedef struct SizeParams
{
  int batch;
  int inputs;
  int h;
  int w;
  int c;
  int index;
  int time_steps;
  int train;
  Network* net;
} SizeParams;

void ParseLocal(layer* l, list* options, SizeParams params)
{
  int n = FindOptionInt(options, "filters", 1);
  int size = FindOptionInt(options, "size", 1);
  int stride = FindOptionInt(options, "stride", 1);
  int pad = FindOptionInt(options, "pad", 0);
  char* activation_str = FindOptionStr(options, "activation", "logistic");
  ACTIVATION activation = get_activation(activation_str);

  int h = params.h;
  int w = params.w;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before local layer must output image");

  FillLocalLayer(l, params.batch, h, w, c, n, size, stride, pad, activation);
}

void ParseConv(layer* l, list* options, SizeParams params)
{
  int n = FindOptionInt(options, "filters", 1);
  int groups = FindOptionIntQuiet(options, "groups", 1);
  int size = FindOptionInt(options, "size", 1);
  int stride = -1;
  int stride_x = FindOptionIntQuiet(options, "stride_x", -1);
  int stride_y = FindOptionIntQuiet(options, "stride_y", -1);
  if (stride_x < 1 || stride_y < 1)
  {
    stride = FindOptionInt(options, "stride", 1);
    if (stride_x < 1)
      stride_x = stride;
    if (stride_y < 1)
      stride_y = stride;
  }
  else
  {
    stride = FindOptionIntQuiet(options, "stride", 1);
  }
  int dilation = FindOptionIntQuiet(options, "dilation", 1);
  int antialiasing = FindOptionIntQuiet(options, "antialiasing", 0);
  if (size == 1)
    dilation = 1;
  int pad = FindOptionIntQuiet(options, "pad", 0);
  int padding = FindOptionIntQuiet(options, "padding", 0);
  if (pad)
    padding = size / 2;

  char* activation_str = FindOptionStr(options, "activation", "logistic");
  ACTIVATION activation = get_activation(activation_str);

  int share_index = FindOptionIntQuiet(options, "share_index", -1000000000);
  layer* share_layer = NULL;
  if (share_index >= 0)
    share_layer = &params.net->layers[share_index];
  else if (share_index != -1000000000)
    share_layer = &params.net->layers[params.index + share_index];

  int h = params.h;
  int w = params.w;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before convolutional layer must output image.");

  int batch_normalize = FindOptionIntQuiet(options, "batch_normalize", 0);
  int binary = FindOptionIntQuiet(options, "binary", 0);
  int xnor = FindOptionIntQuiet(options, "xnor", 0);
  int use_bin_output = FindOptionIntQuiet(options, "bin_output", 0);

  FillConvLayer(l, params.batch, 1, h, w, c, n, groups, size, stride_x,
      stride_y, dilation, padding, activation, batch_normalize, binary, xnor,
      params.net->adam, use_bin_output, params.index, antialiasing, share_layer,
      params.train);

  l->angle = FindOptionFloatQuiet(options, "angle", 15);

  if (params.net->adam)
  {
    l->B1 = params.net->B1;
    l->B2 = params.net->B2;
    l->eps = params.net->eps;
  }
}

void ParseConnected(layer* l, list* options, SizeParams params)
{
  int output = FindOptionInt(options, "output", 1);
  char* activation_str = FindOptionStr(options, "activation", "logistic");
  ACTIVATION activation = get_activation(activation_str);
  int batch_normalize = FindOptionIntQuiet(options, "batch_normalize", 0);

  FillConnectedLayer(
      l, params.batch, 1, params.inputs, output, activation, batch_normalize);
}

void ParseSoftmax(layer* l, list* options, SizeParams params)
{
  int groups = FindOptionIntQuiet(options, "groups", 1);

  FillSoftmaxLayer(l, params.batch, params.inputs, groups);

  l->temperature = FindOptionFloatQuiet(options, "temperature", 1);
  char* tree_file = FindOptionStr(options, "tree", 0);
  if (tree_file)
    l->softmax_tree = read_tree(tree_file);
  l->w = params.w;
  l->h = params.h;
  l->c = params.c;
  l->spatial = FindOptionFloatQuiet(options, "spatial", 0);
  l->noloss = FindOptionIntQuiet(options, "noloss", 0);
}

int* parse_yolo_mask(char* a, int* num)
{
  int* mask = 0;
  if (a)
  {
    int len = strlen(a);
    int n = 1;
    int i;
    for (i = 0; i < len; ++i)
    {
      if (a[i] == ',')
        ++n;
    }
    mask = (int*)xcalloc(n, sizeof(int));
    for (i = 0; i < n; ++i)
    {
      int val = atoi(a);
      mask[i] = val;
      a = strchr(a, ',') + 1;
    }
    *num = n;
  }
  return mask;
}

float* get_classes_multipliers(char* cpc, const int classes)
{
  float* classes_multipliers = NULL;
  if (cpc)
  {
    int classes_counters = classes;
    int* counters_per_class = parse_yolo_mask(cpc, &classes_counters);
    if (classes_counters != classes)
    {
      printf(
          " number of values in counters_per_class = %d doesn't match with "
          "classes = %d \n",
          classes_counters, classes);
      exit(0);
    }
    float max_counter = 0;
    int i;
    for (i = 0; i < classes_counters; ++i)
      if (max_counter < counters_per_class[i])
        max_counter = counters_per_class[i];
    classes_multipliers = (float*)calloc(classes_counters, sizeof(float));
    for (i = 0; i < classes_counters; ++i)
      classes_multipliers[i] = max_counter / counters_per_class[i];
    free(counters_per_class);
    printf(" classes_multipliers: ");
    for (i = 0; i < classes_counters; ++i)
      printf("%.1f, ", classes_multipliers[i]);
    printf("\n");
  }
  return classes_multipliers;
}

void ParseYolo(layer* l, list* options, SizeParams params)
{
  int classes = FindOptionInt(options, "classes", 20);
  int total = FindOptionInt(options, "num", 1);
  int num = total;
  char* a = FindOptionStr(options, "mask", 0);
  int* mask = parse_yolo_mask(a, &num);
  int max_boxes = FindOptionIntQuiet(options, "max", 90);

  FillYoloLayer(l, params.batch, params.w, params.h, num, total, mask, classes,
      max_boxes);

  if (l->outputs != params.inputs)
  {
    printf("Error: l->outputs == params.inputs \n");
    printf(
        "filters= in the [convolutional]-layer doesn't correspond to classes= "
        "or mask= in [yolo]-layer \n");
    exit(EXIT_FAILURE);
  }

  char* cpc = FindOptionStr(options, "counters_per_class", 0);
  l->classes_multipliers = get_classes_multipliers(cpc, classes);

  l->label_smooth_eps = FindOptionFloatQuiet(options, "label_smooth_eps", 0.0f);
  l->scale_x_y = FindOptionFloatQuiet(options, "scale_x_y", 1);
  l->max_delta = FindOptionFloatQuiet(options, "max_delta", FLT_MAX);  // set 10
  l->iou_normalizer = FindOptionFloatQuiet(options, "iou_normalizer", 0.75);
  l->cls_normalizer = FindOptionFloatQuiet(options, "cls_normalizer", 1);
  char* iou_loss = FindOptionStrQuiet(options, "iou_loss", "mse");  //  "iou");

  if (strcmp(iou_loss, "mse") == 0)
    l->iou_loss = MSE;
  else if (strcmp(iou_loss, "giou") == 0)
    l->iou_loss = GIOU;
  else if (strcmp(iou_loss, "diou") == 0)
    l->iou_loss = DIOU;
  else if (strcmp(iou_loss, "ciou") == 0)
    l->iou_loss = CIOU;
  else
    l->iou_loss = IOU;
  fprintf(stderr,
      "[yolo] params: iou loss: %s (%d), iou_norm: %2.2f, cls_norm: %2.2f, "
      "scale_x_y: %2.2f\n",
      iou_loss, l->iou_loss, l->iou_normalizer, l->cls_normalizer,
      l->scale_x_y);

  char* iou_thresh_kind_str =
      FindOptionStrQuiet(options, "iou_thresh_kind", "iou");
  if (strcmp(iou_thresh_kind_str, "iou") == 0)
    l->iou_thresh_kind = IOU;
  else if (strcmp(iou_thresh_kind_str, "giou") == 0)
    l->iou_thresh_kind = GIOU;
  else if (strcmp(iou_thresh_kind_str, "diou") == 0)
    l->iou_thresh_kind = DIOU;
  else if (strcmp(iou_thresh_kind_str, "ciou") == 0)
    l->iou_thresh_kind = CIOU;
  else
  {
    fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
    l->iou_thresh_kind = IOU;
  }

  l->beta_nms = FindOptionFloatQuiet(options, "beta_nms", 0.6);
  char* nms_kind = FindOptionStrQuiet(options, "nms_kind", "default");
  if (strcmp(nms_kind, "default") == 0)
  {
    l->nms_kind = DEFAULT_NMS;
  }
  else
  {
    if (strcmp(nms_kind, "greedynms") == 0)
      l->nms_kind = GREEDY_NMS;
    else if (strcmp(nms_kind, "diounms") == 0)
      l->nms_kind = DIOU_NMS;
    else
      l->nms_kind = DEFAULT_NMS;
  }
  printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l->nms_kind, l->beta_nms);

  l->jitter = FindOptionFloat(options, "jitter", .2);
  l->focal_loss = FindOptionIntQuiet(options, "focal_loss", 0);

  l->ignore_thresh = FindOptionFloat(options, "ignore_thresh", .5);
  l->truth_thresh = FindOptionFloat(options, "truth_thresh", 1);
  l->iou_thresh = FindOptionFloatQuiet(options, "iou_thresh",
      1);  // recommended to use iou_thresh=0.213 in [yolo]
  l->random = FindOptionFloatQuiet(options, "random", 0);

  char* map_file = FindOptionStr(options, "map", 0);
  if (map_file)
    l->map = read_map(map_file);

  a = FindOptionStr(options, "anchors", 0);
  if (a)
  {
    int len = strlen(a);
    int n = 1;
    for (int i = 0; i < len; ++i)
    {
      if (a[i] == ',')
        ++n;
    }
    for (int i = 0; i < n && i < total * 2; ++i)
    {
      float bias = atof(a);
      l->biases[i] = bias;
      a = strchr(a, ',') + 1;
    }
  }
}

int* parse_gaussian_yolo_mask(char* a, int* num)  // Gaussian_YOLOv3
{
  int* mask = 0;
  if (a)
  {
    int len = strlen(a);
    int n = 1;
    int i;
    for (i = 0; i < len; ++i)
    {
      if (a[i] == ',')
        ++n;
    }
    mask = (int*)calloc(n, sizeof(int));
    for (i = 0; i < n; ++i)
    {
      int val = atoi(a);
      mask[i] = val;
      a = strchr(a, ',') + 1;
    }
    *num = n;
  }
  return mask;
}

// Gaussian_YOLOv3
void ParseGaussianYolo(layer* l, list* options, SizeParams params)
{
  int classes = FindOptionInt(options, "classes", 20);
  int max_boxes = FindOptionIntQuiet(options, "max", 90);
  int total = FindOptionInt(options, "num", 1);
  int num = total;

  char* a = FindOptionStr(options, "mask", 0);
  int* mask = parse_gaussian_yolo_mask(a, &num);

  FillGaussianYoloLayer(l, params.batch, params.w, params.h, num, total, mask,
      classes, max_boxes);

  if (l->outputs != params.inputs)
  {
    printf("Error: l->outputs == params.inputs \n");
    printf(
        "filters= in the [convolutional]-layer doesn't correspond to classes= "
        "or mask= in [Gaussian_yolo]-layer \n");
    exit(EXIT_FAILURE);
  }

  char* cpc = FindOptionStr(options, "counters_per_class", 0);
  l->classes_multipliers = get_classes_multipliers(cpc, classes);

  l->label_smooth_eps = FindOptionFloatQuiet(options, "label_smooth_eps", 0.0f);
  l->scale_x_y = FindOptionFloatQuiet(options, "scale_x_y", 1);
  l->max_delta = FindOptionFloatQuiet(options, "max_delta", FLT_MAX);  // set 10
  l->uc_normalizer = FindOptionFloatQuiet(options, "uc_normalizer", 1.0);
  l->iou_normalizer = FindOptionFloatQuiet(options, "iou_normalizer", 0.75);
  l->cls_normalizer = FindOptionFloatQuiet(options, "cls_normalizer", 1.0);
  char* iou_loss = FindOptionStrQuiet(options, "iou_loss", "mse");  //  "iou");

  if (strcmp(iou_loss, "mse") == 0)
    l->iou_loss = MSE;
  else if (strcmp(iou_loss, "giou") == 0)
    l->iou_loss = GIOU;
  else if (strcmp(iou_loss, "diou") == 0)
    l->iou_loss = DIOU;
  else if (strcmp(iou_loss, "ciou") == 0)
    l->iou_loss = CIOU;
  else
    l->iou_loss = IOU;

  char* iou_thresh_kind_str =
      FindOptionStrQuiet(options, "iou_thresh_kind", "iou");
  if (strcmp(iou_thresh_kind_str, "iou") == 0)
    l->iou_thresh_kind = IOU;
  else if (strcmp(iou_thresh_kind_str, "giou") == 0)
    l->iou_thresh_kind = GIOU;
  else if (strcmp(iou_thresh_kind_str, "diou") == 0)
    l->iou_thresh_kind = DIOU;
  else if (strcmp(iou_thresh_kind_str, "ciou") == 0)
    l->iou_thresh_kind = CIOU;
  else
  {
    fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
    l->iou_thresh_kind = IOU;
  }

  l->beta_nms = FindOptionFloatQuiet(options, "beta_nms", 0.6);
  char* nms_kind = FindOptionStrQuiet(options, "nms_kind", "default");
  if (strcmp(nms_kind, "default") == 0)
  {
    l->nms_kind = DEFAULT_NMS;
  }
  else
  {
    if (strcmp(nms_kind, "greedynms") == 0)
      l->nms_kind = GREEDY_NMS;
    else if (strcmp(nms_kind, "diounms") == 0)
      l->nms_kind = DIOU_NMS;
    else if (strcmp(nms_kind, "cornersnms") == 0)
      l->nms_kind = CORNERS_NMS;
    else
      l->nms_kind = DEFAULT_NMS;
  }
  printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l->nms_kind, l->beta_nms);

  char* yolo_point = FindOptionStrQuiet(options, "yolo_point", "center");
  if (strcmp(yolo_point, "left_top") == 0)
    l->yolo_point = YOLO_LEFT_TOP;
  else if (strcmp(yolo_point, "right_bottom") == 0)
    l->yolo_point = YOLO_RIGHT_BOTTOM;
  else
    l->yolo_point = YOLO_CENTER;

  fprintf(stderr,
      "[Gaussian_yolo] iou loss: %s (%d), iou_norm: %2.2f, cls_norm: "
      "%2.2f, scale: %2.2f, point: %d\n",
      iou_loss, l->iou_loss, l->iou_normalizer, l->cls_normalizer, l->scale_x_y,
      l->yolo_point);

  l->jitter = FindOptionFloat(options, "jitter", .2);

  l->ignore_thresh = FindOptionFloat(options, "ignore_thresh", .5);
  l->truth_thresh = FindOptionFloat(options, "truth_thresh", 1);
  l->iou_thresh = FindOptionFloatQuiet(options, "iou_thresh",
      1);  // recommended to use iou_thresh=0.213 in [yolo]
  l->random = FindOptionFloatQuiet(options, "random", 0);

  char* map_file = FindOptionStr(options, "map", 0);
  if (map_file)
    l->map = read_map(map_file);

  a = FindOptionStr(options, "anchors", 0);
  if (a)
  {
    int len = strlen(a);
    int n = 1;
    for (int i = 0; i < len; ++i)
    {
      if (a[i] == ',')
        ++n;
    }
    for (int i = 0; i < n; ++i)
    {
      float bias = atof(a);
      l->biases[i] = bias;
      a = strchr(a, ',') + 1;
    }
  }
}

void ParseRegion(layer* l, list* options, SizeParams params)
{
  int coords = FindOptionInt(options, "coords", 4);
  int classes = FindOptionInt(options, "classes", 20);
  int num = FindOptionInt(options, "num", 1);
  int max_boxes = FindOptionIntQuiet(options, "max", 90);

  FillRegionLayer(
      l, params.batch, params.w, params.h, num, classes, coords, max_boxes);

  if (l->outputs != params.inputs)
  {
    printf("Error: l->outputs == params.inputs \n");
    printf(
        "filters= in the [convolutional]-layer doesn't correspond to classes= "
        "or num= in [region]-layer \n");
    exit(EXIT_FAILURE);
  }

  l->log = FindOptionIntQuiet(options, "log", 0);
  l->sqrt = FindOptionIntQuiet(options, "sqrt", 0);

  l->softmax = FindOptionInt(options, "softmax", 0);
  l->focal_loss = FindOptionIntQuiet(options, "focal_loss", 0);
  l->jitter = FindOptionFloat(options, "jitter", .2);
  l->rescore = FindOptionIntQuiet(options, "rescore", 0);

  l->thresh = FindOptionFloat(options, "thresh", .5);
  l->classfix = FindOptionIntQuiet(options, "classfix", 0);
  l->absolute = FindOptionIntQuiet(options, "absolute", 0);
  l->random = FindOptionFloatQuiet(options, "random", 0);

  l->coord_scale = FindOptionFloat(options, "coord_scale", 1);
  l->object_scale = FindOptionFloat(options, "object_scale", 1);
  l->noobject_scale = FindOptionFloat(options, "noobject_scale", 1);
  l->mask_scale = FindOptionFloat(options, "mask_scale", 1);
  l->class_scale = FindOptionFloat(options, "class_scale", 1);
  l->bias_match = FindOptionIntQuiet(options, "bias_match", 0);

  char* tree_file = FindOptionStr(options, "tree", 0);
  if (tree_file)
    l->softmax_tree = read_tree(tree_file);

  char* map_file = FindOptionStr(options, "map", 0);
  if (map_file)
    l->map = read_map(map_file);

  char* a = FindOptionStr(options, "anchors", 0);
  if (a)
  {
    int len = strlen(a);
    int n = 1;
    for (int i = 0; i < len; ++i)
    {
      if (a[i] == ',')
        ++n;
    }
    for (int i = 0; i < n && i < num * 2; ++i)
    {
      float bias = atof(a);
      l->biases[i] = bias;
      a = strchr(a, ',') + 1;
    }
  }
}

void ParseDetection(layer* l, list* options, SizeParams params)
{
  int coords = FindOptionInt(options, "coords", 1);
  int classes = FindOptionInt(options, "classes", 1);
  int rescore = FindOptionInt(options, "rescore", 0);
  int num = FindOptionInt(options, "num", 1);
  int side = FindOptionInt(options, "side", 7);

  FillDetectionLayer(
      l, params.batch, params.inputs, num, side, classes, coords, rescore);

  l->softmax = FindOptionInt(options, "softmax", 0);
  l->sqrt = FindOptionInt(options, "sqrt", 0);

  l->max_boxes = FindOptionIntQuiet(options, "max", 30);
  l->coord_scale = FindOptionFloat(options, "coord_scale", 1);
  l->forced = FindOptionInt(options, "forced", 0);
  l->object_scale = FindOptionFloat(options, "object_scale", 1);
  l->noobject_scale = FindOptionFloat(options, "noobject_scale", 1);
  l->class_scale = FindOptionFloat(options, "class_scale", 1);
  l->jitter = FindOptionFloat(options, "jitter", .2);
  l->random = FindOptionFloatQuiet(options, "random", 0);
  l->reorg = FindOptionIntQuiet(options, "reorg", 0);
}

void ParseCost(layer* l, list* options, SizeParams params)
{
  char* type_str = FindOptionStr(options, "type", "sse");
  float scale = FindOptionFloatQuiet(options, "scale", 1);

  FillCostLayer(l, params.batch, params.inputs, type_str, scale);
  l->ratio = FindOptionFloatQuiet(options, "ratio", 0);
}

void ParseCrop(layer* l, list* options, SizeParams params)
{
  int crop_height = FindOptionInt(options, "crop_height", 1);
  int crop_width = FindOptionInt(options, "crop_width", 1);
  int flip = FindOptionInt(options, "flip", 0);
  float angle = FindOptionFloat(options, "angle", 0);
  float saturation = FindOptionFloat(options, "saturation", 1);
  float exposure = FindOptionFloat(options, "exposure", 1);

  int h = params.h;
  int w = params.w;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before crop layer must output image.");

  FillCropLayer(l, params.batch, h, w, c, crop_height, crop_width, flip, angle,
      saturation, exposure);
  l->shift = FindOptionFloat(options, "shift", 0);
  l->noadjust = FindOptionIntQuiet(options, "noadjust", 0);
}

void ParseReorg(layer* l, list* options, SizeParams params)
{
  int stride = FindOptionInt(options, "stride", 1);
  int reverse = FindOptionIntQuiet(options, "reverse", 0);

  int h = params.h;
  int w = params.w;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before reorg layer must output image.");

  FillReorgLayer(l, params.batch, w, h, c, stride, reverse);
}

void ParseReorgOld(layer* l, list* options, SizeParams params)
{
  int stride = FindOptionInt(options, "stride", 1);
  int reverse = FindOptionIntQuiet(options, "reverse", 0);

  int h = params.h;
  int w = params.w;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before reorg layer must output image.");

  FillReorgOldLayer(l, params.batch, w, h, c, stride, reverse);
}

void ParseMaxpool(layer* l, list* options, SizeParams params)
{
  int stride = FindOptionInt(options, "stride", 1);
  int stride_x = FindOptionIntQuiet(options, "stride_x", stride);
  int stride_y = FindOptionIntQuiet(options, "stride_y", stride);
  int size = FindOptionInt(options, "size", stride);
  int padding = FindOptionIntQuiet(options, "padding", size - 1);
  int maxpool_depth = FindOptionIntQuiet(options, "maxpool_depth", 0);
  int out_channels = FindOptionIntQuiet(options, "out_channels", 1);
  int antialiasing = FindOptionIntQuiet(options, "antialiasing", 0);

  int h = params.h;
  int w = params.w;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before [maxpool] layer must output image.");

  FillMaxpoolLayer(l, params.batch, h, w, c, size, stride_x, stride_y, padding,
      maxpool_depth, out_channels, antialiasing, params.train);
}

void ParseAvgpool(layer* l, list* options, SizeParams params)
{
  int w = params.w;
  int h = params.h;
  int c = params.c;
  if (!(h && w && c))
    error("Layer before avgpool layer must output image.");

  FillAvgpoolLayer(l, params.batch, w, h, c);
}

void ParseDropout(layer* l, list* options, SizeParams params)
{
  float probability = FindOptionFloat(options, "probability", .2);
  int dropblock = FindOptionIntQuiet(options, "dropblock", 0);
  float dropblock_size_rel =
      FindOptionFloatQuiet(options, "dropblock_size_rel", 0);
  int dropblock_size_abs =
      FindOptionFloatQuiet(options, "dropblock_size_abs", 0);

  if (dropblock_size_abs > params.w || dropblock_size_abs > params.h)
  {
    printf(
        " [dropout] - dropblock_size_abs = %d that is bigger than layer size "
        "%d x %d \n",
        dropblock_size_abs, params.w, params.h);
    dropblock_size_abs = min_val_cmp(params.w, params.h);
  }

  if (dropblock && !dropblock_size_rel && !dropblock_size_abs)
  {
    printf(
        " [dropout] - None of the parameters (dropblock_size_rel or "
        "dropblock_size_abs) are set, will be used: dropblock_size_abs = 7 \n");
    dropblock_size_abs = 7;
  }

  if (dropblock_size_rel && dropblock_size_abs)
  {
    printf(
        " [dropout] - Both parameters are set, only the parameter will be "
        "used: dropblock_size_abs = %d \n",
        dropblock_size_abs);
    dropblock_size_rel = 0;
  }

  FillDropoutLayer(l, params.batch, params.inputs, probability, dropblock,
      dropblock_size_rel, dropblock_size_abs, params.w, params.h, params.c);
  l->out_w = params.w;
  l->out_h = params.h;
  l->out_c = params.c;
}

void ParseBatchnorm(layer* l, list* options, SizeParams params)
{
  FillBatchnormLayer(
      l, params.batch, params.w, params.h, params.c, params.train);
}

void ParseShortcut(layer* l, list* options, SizeParams params, Network* net)
{
  char* activation_str = FindOptionStr(options, "activation", "linear");
  ACTIVATION activation = get_activation(activation_str);

  char* weights_type_str = FindOptionStrQuiet(options, "weights_type", "none");
  WEIGHTS_TYPE_T weights_type = NO_WEIGHTS;
  if (strcmp(weights_type_str, "per_feature") == 0 ||
      strcmp(weights_type_str, "per_layer") == 0)
    weights_type = PER_FEATURE;
  else if (strcmp(weights_type_str, "per_channel") == 0)
    weights_type = PER_CHANNEL;
  else if (strcmp(weights_type_str, "none") != 0)
  {
    printf(
        "Error: Incorrect weights_type = %s \n Use one of: none, per_feature, "
        "per_channel \n",
        weights_type_str);
    getchar();
    exit(0);
  }

  char* weights_normalizion_str =
      FindOptionStrQuiet(options, "weights_normalization", "none");
  WEIGHTS_NORMALIZATION_T weights_normalization = NO_NORMALIZATION;
  if (strcmp(weights_normalizion_str, "relu") == 0 ||
      strcmp(weights_normalizion_str, "avg_relu") == 0)
    weights_normalization = RELU_NORMALIZATION;
  else if (strcmp(weights_normalizion_str, "softmax") == 0)
    weights_normalization = SOFTMAX_NORMALIZATION;
  else if (strcmp(weights_type_str, "none") != 0)
  {
    printf(
        "Error: Incorrect weights_normalization = %s \n Use one of: none, "
        "relu, "
        "softmax \n",
        weights_normalizion_str);
    getchar();
    exit(0);
  }

  char* from = FindOption(options, "from");
  if (!from)
    error("Route Layer must specify input layers: from = ...");

  int n = 1;
  for (int i = 0; i < strlen(from); ++i)
  {
    if (from[i] == ',')
      ++n;
  }

  int* layers = (int*)calloc(n, sizeof(int));
  int* sizes = (int*)calloc(n, sizeof(int));
  float** layers_output = (float**)calloc(n, sizeof(float*));
  float** layers_delta = (float**)calloc(n, sizeof(float*));
  float** layers_output_gpu = (float**)calloc(n, sizeof(float*));
  float** layers_delta_gpu = (float**)calloc(n, sizeof(float*));

  for (int i = 0; i < n; ++i)
  {
    int idx = atoi(from);
    from = strchr(from, ',') + 1;
    if (idx < 0)
      idx = params.index + idx;
    layers[i] = idx;
    sizes[i] = params.net->layers[idx].outputs;
    layers_output[i] = params.net->layers[idx].output;
    layers_delta[i] = params.net->layers[idx].delta;
  }

#ifdef GPU
  for (int i = 0; i < n; ++i)
  {
    layers_output_gpu[i] = params.net->layers[layers[i]].output_gpu;
    layers_delta_gpu[i] = params.net->layers[layers[i]].delta_gpu;
  }
#endif  // GPU

  FillShortcutLayer(l, params.batch, n, layers, sizes, params.w, params.h,
      params.c, layers_output, layers_delta, layers_output_gpu,
      layers_delta_gpu, weights_type, weights_normalization, activation,
      params.train);

  free(layers_output_gpu);
  free(layers_delta_gpu);

  for (int i = 0; i < n; ++i)
  {
    int idx = layers[i];
    assert(params.w == net->layers[idx].out_w &&
           params.h == net->layers[idx].out_h);

    if (params.w != net->layers[idx].out_w ||
        params.h != net->layers[idx].out_h ||
        params.c != net->layers[idx].out_c)
      fprintf(stderr, " (%4d x%4d x%4d) + (%4d x%4d x%4d) \n", params.w,
          params.h, params.c, net->layers[idx].out_w, net->layers[idx].out_h,
          params.net->layers[idx].out_c);
  }
}

void ParseScaleChannels(
    layer* l, list* options, SizeParams params, Network* net)
{
  char* from = FindOption(options, "from");
  int idx = atoi(from);
  if (idx < 0)
    idx = params.index + idx;
  int scale_wh = FindOptionIntQuiet(options, "scale_wh", 0);

  layer* from_l = &net->layers[idx];
  FillScaleChannelsLayer(l, params.batch, idx, params.w, params.h, params.c,
      from_l->out_w, from_l->out_h, from_l->out_c, scale_wh);

  char* activation_str = FindOptionStrQuiet(options, "activation", "linear");
  ACTIVATION activation = get_activation(activation_str);
  l->activation = activation;
  if (activation == SWISH || activation == MISH)
  {
    printf(
        " [scale_channels] layer doesn't support SWISH or MISH activations \n");
  }
}

void ParseActivation(layer* l, list* options, SizeParams params)
{
  char* activation_str = FindOptionStr(options, "activation", "linear");
  ACTIVATION activation = get_activation(activation_str);

  FillActivationLayer(l, params.batch, params.inputs, activation);

  l->out_h = params.h;
  l->out_w = params.w;
  l->out_c = params.c;
  l->h = params.h;
  l->w = params.w;
  l->c = params.c;
}

void ParseUpsample(layer* l, list* options, SizeParams params)
{
  int stride = FindOptionInt(options, "stride", 2);

  FillUpsampleLayer(l, params.batch, params.w, params.h, params.c, stride);

  l->scale = FindOptionFloatQuiet(options, "scale", 1);
}

void ParseRoute(layer* l, list* options, SizeParams params)
{
  char* input_layers = FindOption(options, "layers");
  if (!input_layers)
    error("Route Layer must specify input layers");
  int len = strlen(input_layers);
  int n = 1;
  for (int i = 0; i < len; ++i)
  {
    if (input_layers[i] == ',')
      ++n;
  }

  int* layers = (int*)xcalloc(n, sizeof(int));
  int* sizes = (int*)xcalloc(n, sizeof(int));
  for (int i = 0; i < n; ++i)
  {
    int index = atoi(input_layers);
    input_layers = strchr(input_layers, ',') + 1;
    if (index < 0)
      index = params.index + index;
    layers[i] = index;
    sizes[i] = params.net->layers[index].outputs;
  }

  int groups = FindOptionIntQuiet(options, "groups", 1);
  int group_id = FindOptionIntQuiet(options, "group_id", 0);

  FillRouteLayer(l, params.batch, n, layers, sizes, groups, group_id);

  layer first = params.net->layers[layers[0]];
  l->out_w = first.out_w;
  l->out_h = first.out_h;
  l->out_c = first.out_c;
  for (int i = 1; i < n; ++i)
  {
    int index = layers[i];
    layer next = params.net->layers[index];
    if (next.out_w == first.out_w && next.out_h == first.out_h)
    {
      l->out_c += next.out_c;
    }
    else
    {
      fprintf(stderr,
          " The width and height of the input layers are different. \n");
      l->out_h = l->out_w = l->out_c = 0;
    }
  }
  l->out_c = l->out_c / l->groups;

  l->w = first.w;
  l->h = first.h;
  l->c = l->out_c;

  if (n > 3)
    fprintf(stderr, " \t    ");
  else if (n > 1)
    fprintf(stderr, " \t            ");
  else
    fprintf(stderr, " \t\t            ");

  fprintf(stderr, "           ");
  if (l->groups > 1)
    fprintf(stderr, "%d/%d", l->group_id, l->groups);
  else
    fprintf(stderr, "   ");
  fprintf(stderr, " -> %4d x%4d x%4d \n", l->out_w, l->out_h, l->out_c);
}

LearningRatePolicy GetPolicy(char* s)
{
  if (strcmp(s, "random") == 0)
    return RANDOM;
  if (strcmp(s, "poly") == 0)
    return POLY;
  if (strcmp(s, "constant") == 0)
    return CONSTANT;
  if (strcmp(s, "step") == 0)
    return STEP;
  if (strcmp(s, "exp") == 0)
    return EXP;
  if (strcmp(s, "sigmoid") == 0)
    return SIG;
  if (strcmp(s, "steps") == 0)
    return STEPS;
  if (strcmp(s, "sgdr") == 0)
    return SGDR;

  fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
  return CONSTANT;
}

void ParseNetOptions(list* options, Network* net)
{
  net->max_batches = FindOptionInt(options, "max_batches", 0);
  net->batch = FindOptionInt(options, "batch", 1);
  net->learning_rate = FindOptionFloat(options, "learning_rate", .001);
  net->learning_rate_min =
      FindOptionFloatQuiet(options, "learning_rate_min", .00001);
  net->batches_per_cycle =
      FindOptionIntQuiet(options, "sgdr_cycle", net->max_batches);
  net->batches_cycle_mult = FindOptionIntQuiet(options, "sgdr_mult", 2);
  net->momentum = FindOptionFloat(options, "momentum", .9);
  net->decay = FindOptionFloat(options, "decay", .0001);
  int subdivs = FindOptionInt(options, "subdivisions", 1);
  net->time_steps = FindOptionIntQuiet(options, "time_steps", 1);
  net->track = FindOptionIntQuiet(options, "track", 0);
  net->augment_speed = FindOptionIntQuiet(options, "augment_speed", 2);
  net->init_seq_subdiv = net->seq_subdiv =
      FindOptionIntQuiet(options, "sequential_subdivisions", subdivs);
  if (net->seq_subdiv > subdivs)
    net->init_seq_subdiv = net->seq_subdiv = subdivs;
  net->try_fix_nan = FindOptionIntQuiet(options, "try_fix_nan", 0);
  net->batch /= subdivs;
  net->batch *= net->time_steps;
  net->subdiv = subdivs;

  *net->seen = 0;
  *net->cur_iteration = 0;
  net->loss_scale = FindOptionFloatQuiet(options, "loss_scale", 1);
  net->optimized_memory = FindOptionIntQuiet(options, "optimized_memory", 0);
  net->workspace_size_limit =
      (size_t)1024 * 1024 *
      FindOptionFloatQuiet(options, "workspace_size_limit_MB",
          1024);  // 1024 MB by default

  net->adam = FindOptionIntQuiet(options, "adam", 0);
  if (net->adam)
  {
    net->B1 = FindOptionFloat(options, "B1", .9);
    net->B2 = FindOptionFloat(options, "B2", .999);
    net->eps = FindOptionFloat(options, "eps", .000001);
  }

  net->h = FindOptionIntQuiet(options, "height", 0);
  net->w = FindOptionIntQuiet(options, "width", 0);
  net->c = FindOptionIntQuiet(options, "channels", 0);
  net->inputs = FindOptionIntQuiet(options, "inputs", net->h * net->w * net->c);
  net->max_crop = FindOptionIntQuiet(options, "max_crop", net->w * 2);
  net->min_crop = FindOptionIntQuiet(options, "min_crop", net->w);
  net->flip = FindOptionIntQuiet(options, "flip", 1);
  net->blur = FindOptionIntQuiet(options, "blur", 0);
  net->gaussian_noise = FindOptionIntQuiet(options, "gaussian_noise", 0);
  net->mixup = FindOptionIntQuiet(options, "mixup", 0);
  int cutmix = FindOptionIntQuiet(options, "cutmix", 0);
  int mosaic = FindOptionIntQuiet(options, "mosaic", 0);
  if (mosaic && cutmix)
    net->mixup = 4;
  else if (cutmix)
    net->mixup = 2;
  else if (mosaic)
    net->mixup = 3;
  net->letter_box = FindOptionIntQuiet(options, "letter_box", 0);
  net->label_smooth_eps =
      FindOptionFloatQuiet(options, "label_smooth_eps", 0.0f);
  net->resize_step = FindOptionFloatQuiet(options, "resize_step", 32);
  net->attention = FindOptionIntQuiet(options, "attention", 0);
  net->adversarial_lr = FindOptionFloatQuiet(options, "adversarial_lr", 0);

  net->angle = FindOptionFloatQuiet(options, "angle", 0);
  net->aspect = FindOptionFloatQuiet(options, "aspect", 1);
  net->saturation = FindOptionFloatQuiet(options, "saturation", 1);
  net->exposure = FindOptionFloatQuiet(options, "exposure", 1);
  net->hue = FindOptionFloatQuiet(options, "hue", 0);
  net->power = FindOptionFloatQuiet(options, "power", 4);

  if (!net->inputs && !(net->h && net->w && net->c))
    error("No input parameters supplied");

  char* policy_s = FindOptionStr(options, "policy", "constant");
  net->policy = GetPolicy(policy_s);
  net->burn_in = FindOptionIntQuiet(options, "burn_in", 0);
#ifdef GPU
  if (net->gpu_index >= 0)
  {
    int compute_capability = get_gpu_compute_capability(net->gpu_index);
#ifdef CUDNN_HALF
    if (compute_capability >= 700)
      net->cudnn_half = 1;
    else
      net->cudnn_half = 0;
#endif  // CUDNN_HALF
    fprintf(stderr, " compute_capability = %d, cudnn_half = %d \n",
        compute_capability, net->cudnn_half);
  }
  else
    fprintf(stderr, " GPU isn't used \n");
#endif  // GPU
  if (net->policy == STEP)
  {
    net->step = FindOptionInt(options, "step", 1);
    net->scale = FindOptionFloat(options, "scale", 1);
  }
  else if (net->policy == STEPS || net->policy == SGDR)
  {
    char* l = FindOption(options, "steps");
    char* p = FindOption(options, "scales");
    char* s = FindOption(options, "seq_scales");
    if (net->policy == STEPS && (!l || !p))
      error("STEPS policy must have steps and scales in cfg file");

    if (l)
    {
      int len = strlen(l);
      int n = 1;
      int i;
      for (i = 0; i < len; ++i)
      {
        if (l[i] == ',')
          ++n;
      }
      int* steps = (int*)xcalloc(n, sizeof(int));
      float* scales = (float*)xcalloc(n, sizeof(float));
      float* seq_scales = (float*)xcalloc(n, sizeof(float));
      for (i = 0; i < n; ++i)
      {
        float scale = 1.0;
        if (p)
        {
          scale = atof(p);
          p = strchr(p, ',') + 1;
        }
        float sequence_scale = 1.0;
        if (s)
        {
          sequence_scale = atof(s);
          s = strchr(s, ',') + 1;
        }
        int step = atoi(l);
        l = strchr(l, ',') + 1;
        steps[i] = step;
        scales[i] = scale;
        seq_scales[i] = sequence_scale;
      }
      net->scales = scales;
      net->steps = steps;
      net->seq_scales = seq_scales;
      net->num_steps = n;
    }
  }
  else if (net->policy == EXP)
  {
    net->gamma = FindOptionFloat(options, "gamma", 1);
  }
  else if (net->policy == SIG)
  {
    net->gamma = FindOptionFloat(options, "gamma", 1);
    net->step = FindOptionInt(options, "step", 1);
  }
  else if (net->policy == POLY || net->policy == RANDOM)
  {
    // net->power = option_find_float(options, "power", 1);
  }
}

int IsNetwork(Section* s)
{
  return (strcmp(s->type, "[net]") == 0 || strcmp(s->type, "[network]") == 0);
}

void SetTrainOnlyBn(Network* net)
{
  int train_only_bn = 0;
  for (int i = net->n - 1; i >= 0; --i)
  {
    // set l->train_only_bn for all previous layers
    if (net->layers[i].train_only_bn)
      train_only_bn = net->layers[i].train_only_bn;

    if (train_only_bn)
      net->layers[i].train_only_bn = train_only_bn;
  }
}

void ParseNetworkCfg(Network* net, char const* filename)
{
  ParseNetworkCfgCustom(net, filename, 0, 0);
}

void ParseNetworkCfgCustom(
    Network* net, char const* filename, int batch, int time_steps)
{
  list* sections = ReadSections(filename);
  node* n = sections->front;
  if (!n)
    error("Config file has no sections");

  AllocateNetwork(net, sections->size - 1);
  net->gpu_index = gpu_index;

  SizeParams params;
  if (batch > 0)
    params.train = 0;  // allocates memory for Detection only
  else
    params.train = 1;  // allocates memory for Detection & Training

  Section* s = (Section*)n->val;
  if (!IsNetwork(s))
    error("First section must be [net] or [network]");

  list* options = s->options;
  ParseNetOptions(options, net);

#ifdef GPU
  printf("net->optimized_memory = %d \n", net->optimized_memory);
  if (net->optimized_memory >= 2 && params.train)
  {
    // pre-allocate 8 GB CPU-RAM for pinned memory
    pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 8);
  }
#endif  // GPU

  params.h = net->h;
  params.w = net->w;
  params.c = net->c;
  params.inputs = net->inputs;
  if (batch > 0)
    net->batch = batch;
  if (time_steps > 0)
    net->time_steps = time_steps;
  if (net->batch < 1)
    net->batch = 1;
  if (net->time_steps < 1)
    net->time_steps = 1;
  if (net->batch < net->time_steps)
    net->batch = net->time_steps;
  params.batch = net->batch;
  params.time_steps = net->time_steps;
  params.net = net;
  printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n",
      net->batch, net->batch * net->subdiv, net->time_steps, params.train);

  int avg_outputs = 0;
  int avg_counter = 0;
  float bflops = 0;
  size_t workspace_size = 0;
  int max_inputs = 0;
  int max_outputs = 0;
  int receptive_w = 1, receptive_h = 1;
  int receptive_w_scale = 1, receptive_h_scale = 1;
  int const show_receptive_field =
      FindOptionFloatQuiet(options, "show_receptive_field", 0);

  n = n->next;
  int count = 0;
  FreeSection(s);
  fprintf(stderr,
      "   layer   filters  size/strd(dil)      input                output\n");
  while (n)
  {
    params.index = count;
    fprintf(stderr, "%4d ", count);

    s = (Section*)n->val;
    options = s->options;

    layer* l = &net->layers[count];
    LAYER_TYPE lt = StrToLayerType(s->type);
    if (lt == CONVOLUTIONAL)
    {
      ParseConv(l, options, params);
    }
    else if (lt == LOCAL)
    {
      ParseLocal(l, options, params);
    }
    else if (lt == ACTIVE)
    {
      ParseActivation(l, options, params);
    }
    else if (lt == CONNECTED)
    {
      ParseConnected(l, options, params);
    }
    else if (lt == CROP)
    {
      ParseCrop(l, options, params);
    }
    else if (lt == COST)
    {
      ParseCost(l, options, params);
      l->keep_delta_gpu = 1;
    }
    else if (lt == REGION)
    {
      ParseRegion(l, options, params);
      l->keep_delta_gpu = 1;
    }
    else if (lt == YOLO)
    {
      ParseYolo(l, options, params);
      l->keep_delta_gpu = 1;
    }
    else if (lt == GAUSSIAN_YOLO)
    {
      ParseGaussianYolo(l, options, params);
      l->keep_delta_gpu = 1;
    }
    else if (lt == DETECTION)
    {
      ParseDetection(l, options, params);
    }
    else if (lt == SOFTMAX)
    {
      ParseSoftmax(l, options, params);
      net->hierarchy = l->softmax_tree;
      l->keep_delta_gpu = 1;
    }
    else if (lt == BATCHNORM)
    {
      ParseBatchnorm(l, options, params);
    }
    else if (lt == MAXPOOL)
    {
      ParseMaxpool(l, options, params);
    }
    else if (lt == REORG)
    {
      ParseReorg(l, options, params);
    }
    else if (lt == REORG_OLD)
    {
      ParseReorgOld(l, options, params);
    }
    else if (lt == AVGPOOL)
    {
      ParseAvgpool(l, options, params);
    }
    else if (lt == ROUTE)
    {
      ParseRoute(l, options, params);
      for (int k = 0; k < l->n; ++k)
      {
        net->layers[l->input_layers[k]].use_bin_output = 0;
        net->layers[l->input_layers[k]].keep_delta_gpu = 1;
      }
    }
    else if (lt == UPSAMPLE)
    {
      ParseUpsample(l, options, params);
    }
    else if (lt == SHORTCUT)
    {
      ParseShortcut(l, options, params, net);
      net->layers[count - 1].use_bin_output = 0;
      net->layers[l->index].use_bin_output = 0;
      net->layers[l->index].keep_delta_gpu = 1;
    }
    else if (lt == SCALE_CHANNELS)
    {
      ParseScaleChannels(l, options, params, net);
      net->layers[count - 1].use_bin_output = 0;
      net->layers[l->index].use_bin_output = 0;
      net->layers[l->index].keep_delta_gpu = 1;
    }
    else if (lt == DROPOUT)
    {
      ParseDropout(l, options, params);
      l->output = net->layers[count - 1].output;
      l->delta = net->layers[count - 1].delta;
#ifdef GPU
      l->output_gpu = net->layers[count - 1].output_gpu;
      l->delta_gpu = net->layers[count - 1].delta_gpu;
      l->keep_delta_gpu = 1;
#endif
    }
    else if (lt == EMPTY)
    {
      l->out_w = params.w;
      l->out_h = params.h;
      l->out_c = params.c;
      l->output = net->layers[count - 1].output;
      l->delta = net->layers[count - 1].delta;
#ifdef GPU
      l->output_gpu = net->layers[count - 1].output_gpu;
      l->delta_gpu = net->layers[count - 1].delta_gpu;
#endif
    }
    else
    {
      fprintf(stderr, "Type is not recognized: %s\n", s->type);
    }

    // calculate receptive field
    if (show_receptive_field)
    {
      int dilation = max_val_cmp(1, l->dilation);
      int stride = max_val_cmp(1, l->stride);
      int size = max_val_cmp(1, l->size);

      if (l->type == UPSAMPLE || (l->type == REORG))
      {
        l->receptive_w = receptive_w;
        l->receptive_h = receptive_h;
        l->receptive_w_scale = receptive_w_scale = receptive_w_scale / stride;
        l->receptive_h_scale = receptive_h_scale = receptive_h_scale / stride;
      }
      else
      {
        if (l->type == ROUTE)
        {
          receptive_w = receptive_h = receptive_w_scale = receptive_h_scale = 0;
          for (int k = 0; k < l->n; ++k)
          {
            layer* route_l = &net->layers[l->input_layers[k]];
            receptive_w = max_val_cmp(receptive_w, route_l->receptive_w);
            receptive_h = max_val_cmp(receptive_h, route_l->receptive_h);
            receptive_w_scale =
                max_val_cmp(receptive_w_scale, route_l->receptive_w_scale);
            receptive_h_scale =
                max_val_cmp(receptive_h_scale, route_l->receptive_h_scale);
          }
        }
        else
        {
          int increase_receptive = size + (dilation - 1) * 2 - 1;  // stride;
          increase_receptive = max_val_cmp(0, increase_receptive);

          receptive_w += increase_receptive * receptive_w_scale;
          receptive_h += increase_receptive * receptive_h_scale;
          receptive_w_scale *= stride;
          receptive_h_scale *= stride;
        }

        l->receptive_w = receptive_w;
        l->receptive_h = receptive_h;
        l->receptive_w_scale = receptive_w_scale;
        l->receptive_h_scale = receptive_h_scale;
      }
      // printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d,
      // receptive_w_scale = %d - ", size, dilation, stride, receptive_w,
      // receptive_w_scale);

      int cur_receptive_w = receptive_w;
      int cur_receptive_h = receptive_h;

      fprintf(stderr, "%4d - receptive field: %d x %d \n", count,
          cur_receptive_w, cur_receptive_h);
    }

#ifdef GPU
    // futher GPU-memory optimization: net->optimized_memory == 2
    if (net->optimized_memory >= 2 && params.train && l->type != DROPOUT)
    {
      l->optimized_memory = net->optimized_memory;
      if (l->output_gpu)
      {
        cuda_free(l->output_gpu);
        l->output_gpu =
            cuda_make_array_pinned_preallocated(NULL, l->batch * l->outputs);
      }
      if (l->activation_input_gpu)
      {
        cuda_free(l->activation_input_gpu);
        l->activation_input_gpu =
            cuda_make_array_pinned_preallocated(NULL, l->batch * l->outputs);
      }
      if (l->x_gpu)
      {
        cuda_free(l->x_gpu);
        l->x_gpu =
            cuda_make_array_pinned_preallocated(NULL, l->batch * l->outputs);
      }

      // maximum optimization
      if (net->optimized_memory >= 3 && l->type != DROPOUT)
      {
        if (l->delta_gpu)
        {
          cuda_free(l->delta_gpu);
          // l->delta_gpu = cuda_make_array_pinned_preallocated(NULL,
          // l->batch*l->outputs); // l->steps printf("\n\n PINNED DELTA GPU =
          // %d \n", l->batch*l->outputs);
        }
      }

      if (l->type == CONVOLUTIONAL)
      {
        set_specified_workspace_limit(
            l, net->workspace_size_limit);  // workspace size limit 1 GB
      }
    }
#endif  // GPU

    l->clip = FindOptionFloatQuiet(options, "clip", 0);
    l->onlyforward = FindOptionIntQuiet(options, "onlyforward", 0);
    l->dont_update = FindOptionIntQuiet(options, "dont_update", 0);
    l->burnin_update = FindOptionIntQuiet(options, "burnin_update", 0);
    l->stopbackward = FindOptionIntQuiet(options, "stopbackward", 0);
    l->train_only_bn = FindOptionIntQuiet(options, "train_only_bn", 0);
    l->dontload = FindOptionIntQuiet(options, "dontload", 0);
    l->dontloadscales = FindOptionIntQuiet(options, "dontloadscales", 0);
    l->learning_rate_scale = FindOptionFloatQuiet(options, "learning_rate", 1);
    UnusedOption(options);

    if (l->workspace_size > workspace_size)
      workspace_size = l->workspace_size;
    if (l->inputs > max_inputs)
      max_inputs = l->inputs;
    if (l->outputs > max_outputs)
      max_outputs = l->outputs;

    FreeSection(s);
    n = n->next;
    ++count;

    if (n)
    {
      if (l->antialiasing)
      {
        params.h = l->input_layer->out_h;
        params.w = l->input_layer->out_w;
        params.c = l->input_layer->out_c;
        params.inputs = l->input_layer->outputs;
      }
      else
      {
        params.h = l->out_h;
        params.w = l->out_w;
        params.c = l->out_c;
        params.inputs = l->outputs;
      }
    }
    if (l->bflops > 0)
      bflops += l->bflops;

    if (l->w > 1 && l->h > 1)
    {
      avg_outputs += l->outputs;
      avg_counter++;
    }
  }
  FreeList(sections);

#ifdef GPU
  if (net->optimized_memory && params.train)
  {
    for (int k = 0; k < net->n; ++k)
    {
      layer* l = &net->layers[k];
      // delta GPU-memory optimization: net->optimized_memory == 1
      if (!l->keep_delta_gpu)
      {
        size_t const delta_size = l->batch * l->outputs;
        if (net->max_delta_gpu_size < delta_size)
        {
          net->max_delta_gpu_size = delta_size;
          if (net->global_delta_gpu)
            cuda_free(net->global_delta_gpu);
          if (net->state_delta_gpu)
            cuda_free(net->state_delta_gpu);
          assert(net->max_delta_gpu_size > 0);
          net->global_delta_gpu =
              (float*)cuda_make_array(NULL, net->max_delta_gpu_size);
          net->state_delta_gpu =
              (float*)cuda_make_array(NULL, net->max_delta_gpu_size);
        }
        if (l->delta_gpu)
        {
          if (net->optimized_memory < 3)
            cuda_free(l->delta_gpu);
        }
        l->delta_gpu = net->global_delta_gpu;
      }

      // maximum optimization
      if (net->optimized_memory >= 3 && l->type != DROPOUT)
      {
        if (l->delta_gpu && l->keep_delta_gpu)
        {
          // cuda_free(l->delta_gpu);   // already called above
          l->delta_gpu =
              cuda_make_array_pinned_preallocated(NULL, l->batch * l->outputs);
        }
      }
    }
  }
#endif

  SetTrainOnlyBn(net);  // set l->train_only_bn for all required layers

  net->outputs = GetNetworkOutputSize(net);
  net->output = GetNetworkOutput(net);
  avg_outputs = avg_outputs / avg_counter;
  fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
  fprintf(stderr, "avg_outputs = %d \n", avg_outputs);

#ifdef GPU
  get_cuda_stream();
  get_cuda_memcpy_stream();
  if (gpu_index >= 0)
  {
    int size = GetNetworkInputSize(net) * net->batch;
    net->input_state_gpu = cuda_make_array(0, size);
    if (cudaSuccess == cudaHostAlloc(&net->input_pinned_cpu,
                           size * sizeof(float), cudaHostRegisterMapped))
    {
      net->input_pinned_cpu_flag = 1;
    }
    else
    {
      cudaGetLastError();  // reset CUDA-error
      net->input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
    }

    // pre-allocate memory for inference on Tensor Cores (fp16)
    if (net->cudnn_half)
    {
      *net->max_input16_size = max_inputs;
      CHECK_CUDA(cudaMalloc((void**)net->input16_gpu,
          *net->max_input16_size * sizeof(short)));  // sizeof(half)
      *net->max_output16_size = max_outputs;
      CHECK_CUDA(cudaMalloc((void**)net->output16_gpu,
          *net->max_output16_size * sizeof(short)));  // sizeof(half)
    }

    if (workspace_size)
    {
      fprintf(stderr, " Allocate additional workspace_size = %1.2f MB \n",
          (float)workspace_size / 1000000);
      net->workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
    }
    else
    {
      net->workspace = (float*)xcalloc(1, workspace_size);
    }
  }
#else
  if (workspace_size)
    net->workspace = (float*)xcalloc(1, workspace_size);
#endif

  LAYER_TYPE lt = net->layers[net->n - 1].type;
  if ((net->w % 32 != 0 || net->h % 32 != 0) &&
      (lt == YOLO || lt == REGION || lt == DETECTION))
  {
    printf(
        "\n Warning: width=%d and height=%d in cfg-file must be divisible by "
        "32 for default networks Yolo v1/v2/v3!!! \n\n",
        net->w, net->h);
  }
}

void SaveShortcutWeights(layer* l, FILE* fp)
{
#ifdef GPU
  if (gpu_index >= 0)
  {
    PullShortcutLayer(l);
    printf("\n PullShortcutLayer \n");
  }
#endif
  for (int i = 0; i < l->nweights; ++i)
  {
    printf(" %f, ", l->weight_updates[i]);
  }
  printf(" l->nweights = %d - update \n", l->nweights);
  for (int i = 0; i < l->nweights; ++i)
  {
    printf(" %f, ", l->weights[i]);
  }
  printf(" l->nweights = %d \n\n", l->nweights);

  int num = l->nweights;
  fwrite(l->weights, sizeof(float), num, fp);
}

void SaveConvolutionalWeights(layer* l, FILE* fp)
{
#ifdef GPU
  if (gpu_index >= 0)
  {
    PullConvolutionalLayer(l);
  }
#endif
  int num = l->nweights;
  fwrite(l->biases, sizeof(float), l->n, fp);
  if (l->batch_normalize)
  {
    fwrite(l->scales, sizeof(float), l->n, fp);
    fwrite(l->rolling_mean, sizeof(float), l->n, fp);
    fwrite(l->rolling_variance, sizeof(float), l->n, fp);
  }
  fwrite(l->weights, sizeof(float), num, fp);
}

void SaveBatchnormWeights(layer* l, FILE* fp)
{
#ifdef GPU
  if (gpu_index >= 0)
    PullBatchnormLayer(l);
#endif
  fwrite(l->biases, sizeof(float), l->c, fp);
  fwrite(l->scales, sizeof(float), l->c, fp);
  fwrite(l->rolling_mean, sizeof(float), l->c, fp);
  fwrite(l->rolling_variance, sizeof(float), l->c, fp);
}

void SaveConnectedWeights(layer* l, FILE* fp)
{
#ifdef GPU
  if (gpu_index >= 0)
    PullConnectedLayer(l);
#endif
  fwrite(l->biases, sizeof(float), l->outputs, fp);
  fwrite(l->weights, sizeof(float), l->outputs * l->inputs, fp);
  if (l->batch_normalize)
  {
    fwrite(l->scales, sizeof(float), l->outputs, fp);
    fwrite(l->rolling_mean, sizeof(float), l->outputs, fp);
    fwrite(l->rolling_variance, sizeof(float), l->outputs, fp);
  }
}

void SaveWeightsUpto(Network* net, char* filename, int cutoff)
{
#ifdef GPU
  if (net->gpu_index >= 0)
  {
    cuda_set_device(net->gpu_index);
  }
#endif
  fprintf(stderr, "Saving weights to %s\n", filename);
  FILE* fp = fopen(filename, "wb");
  if (!fp)
    FileError(filename);

  int major = MAJOR_VERSION;
  int minor = MINOR_VERSION;
  int revision = PATCH_VERSION;
  fwrite(&major, sizeof(int), 1, fp);
  fwrite(&minor, sizeof(int), 1, fp);
  fwrite(&revision, sizeof(int), 1, fp);
  (*net->seen) = GetCurrentIteration(net) * net->batch *
                 net->subdiv;  // remove this line, when you will save to
                               // weights-file both: seen & cur_iteration
  fwrite(net->seen, sizeof(uint64_t), 1, fp);

  for (int i = 0; i < net->n && i < cutoff; ++i)
  {
    layer* l = &net->layers[i];
    if (l->type == CONVOLUTIONAL && l->share_layer == NULL)
    {
      SaveConvolutionalWeights(l, fp);
    }
    if (l->type == SHORTCUT && l->nweights > 0)
    {
      SaveShortcutWeights(l, fp);
    }
    if (l->type == CONNECTED)
    {
      SaveConnectedWeights(l, fp);
    }
    if (l->type == BATCHNORM)
    {
      SaveBatchnormWeights(l, fp);
    }
    if (l->type == LOCAL)
    {
#ifdef GPU
      if (gpu_index >= 0)
        PullLocalLayer(l);
#endif
      int locations = l->out_w * l->out_h;
      int size = l->size * l->size * l->c * l->n * locations;
      fwrite(l->biases, sizeof(float), l->outputs, fp);
      fwrite(l->weights, sizeof(float), size, fp);
    }
  }
  fclose(fp);
}

void SaveWeights(Network* net, char* filename)
{
  SaveWeightsUpto(net, filename, net->n);
}

void TransposeMat(float* input, int rows, int cols)
{
  float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
  for (int x = 0; x < rows; ++x)
  {
    for (int y = 0; y < cols; ++y)
    {
      transpose[y * rows + x] = input[x * cols + y];
    }
  }
  memcpy(input, transpose, rows * cols * sizeof(float));
  free(transpose);
}

void LoadConnectedWeights(layer* l, FILE* fp, int transpose)
{
  fread(l->biases, sizeof(float), l->outputs, fp);
  fread(l->weights, sizeof(float), l->outputs * l->inputs, fp);
  if (transpose)
    TransposeMat(l->weights, l->inputs, l->outputs);

  if (l->batch_normalize && (!l->dontloadscales))
  {
    fread(l->scales, sizeof(float), l->outputs, fp);
    fread(l->rolling_mean, sizeof(float), l->outputs, fp);
    fread(l->rolling_variance, sizeof(float), l->outputs, fp);
  }
#ifdef GPU
  if (gpu_index >= 0)
    PushConnectedLayer(l);
#endif
}

void LoadBatchnormWeights(layer* l, FILE* fp)
{
  fread(l->biases, sizeof(float), l->c, fp);
  fread(l->scales, sizeof(float), l->c, fp);
  fread(l->rolling_mean, sizeof(float), l->c, fp);
  fread(l->rolling_variance, sizeof(float), l->c, fp);
#ifdef GPU
  if (gpu_index >= 0)
    PushBatchnormLayer(l);
#endif
}

void LoadConvolutionalWeights(layer* l, FILE* fp)
{
  int num = l->nweights;
  int read_bytes;
  read_bytes = fread(l->biases, sizeof(float), l->n, fp);
  if (read_bytes > 0 && read_bytes < l->n)
    printf(
        "\n Warning: Unexpected end of wights-file! l->biases - l->index = %d "
        "\n",
        l->index);
  // fread(l->weights, sizeof(float), num, fp); // as in connected layer
  if (l->batch_normalize && (!l->dontloadscales))
  {
    read_bytes = fread(l->scales, sizeof(float), l->n, fp);
    if (read_bytes > 0 && read_bytes < l->n)
      printf(
          "\n Warning: Unexpected end of wights-file! l->scales - l->index = "
          "%d "
          "\n",
          l->index);
    read_bytes = fread(l->rolling_mean, sizeof(float), l->n, fp);
    if (read_bytes > 0 && read_bytes < l->n)
      printf(
          "\n Warning: Unexpected end of wights-file! l->rolling_mean - "
          "l->index "
          "= %d \n",
          l->index);
    read_bytes = fread(l->rolling_variance, sizeof(float), l->n, fp);
    if (read_bytes > 0 && read_bytes < l->n)
      printf(
          "\n Warning: Unexpected end of wights-file! l->rolling_variance - "
          "l->index = %d \n",
          l->index);
    if (0)
    {
      int i;
      for (i = 0; i < l->n; ++i)
      {
        printf("%g, ", l->rolling_mean[i]);
      }
      printf("\n");
      for (i = 0; i < l->n; ++i)
      {
        printf("%g, ", l->rolling_variance[i]);
      }
      printf("\n");
    }
    if (0)
    {
      fill_cpu(l->n, 0, l->rolling_mean, 1);
      fill_cpu(l->n, 0, l->rolling_variance, 1);
    }
  }
  read_bytes = fread(l->weights, sizeof(float), num, fp);
  if (read_bytes > 0 && read_bytes < l->n)
    printf(
        "\n Warning: Unexpected end of wights-file! l->weights - l->index = %d "
        "\n",
        l->index);

#ifdef GPU
  if (gpu_index >= 0)
    PushConvolutionalLayer(l);
#endif
}

void LoadShortcutWeights(layer* l, FILE* fp)
{
  int num = l->nweights;
  int read_bytes;
  read_bytes = fread(l->weights, sizeof(float), num, fp);
  if (read_bytes > 0 && read_bytes < num)
    printf(
        "\n Warning: Unexpected end of wights-file! l->weights - l->index = %d "
        "\n",
        l->index);

#ifdef GPU
  if (gpu_index >= 0)
    PushShortcutLayer(l);
#endif
}

void LoadWeightsUpTo(Network* net, char const* filename, int cutoff)
{
#ifdef GPU
  if (net->gpu_index >= 0)
  {
    cuda_set_device(net->gpu_index);
  }
#endif
  fprintf(stderr, "Loading weights from %s...", filename);
  fflush(stdout);
  FILE* fp = fopen(filename, "rb");
  if (!fp)
    FileError(filename);

  int major;
  int minor;
  int revision;
  fread(&major, sizeof(int), 1, fp);
  fread(&minor, sizeof(int), 1, fp);
  fread(&revision, sizeof(int), 1, fp);
  if ((major * 10 + minor) >= 2)
  {
    printf("\n seen 64");
    uint64_t iseen = 0;
    fread(&iseen, sizeof(uint64_t), 1, fp);
    *net->seen = iseen;
  }
  else
  {
    printf("\n seen 32");
    uint32_t iseen = 0;
    fread(&iseen, sizeof(uint32_t), 1, fp);
    *net->seen = iseen;
  }
  *net->cur_iteration = GetCurrentBatch(net);
  printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n",
      (float)(*net->seen / 1000), (float)(*net->seen / 64000));
  int transpose = (major > 1000) || (minor > 1000);

  int i = 0;
  for (i = 0; i < net->n && i < cutoff; ++i)
  {
    layer* l = &net->layers[i];
    if (l->dontload)
      continue;

    if (l->type == CONVOLUTIONAL && l->share_layer == NULL)
    {
      LoadConvolutionalWeights(l, fp);
    }
    if (l->type == SHORTCUT && l->nweights > 0)
    {
      LoadShortcutWeights(l, fp);
    }
    if (l->type == CONNECTED)
    {
      LoadConnectedWeights(l, fp, transpose);
    }
    if (l->type == BATCHNORM)
    {
      LoadBatchnormWeights(l, fp);
    }
    if (l->type == LOCAL)
    {
      int locations = l->out_w * l->out_h;
      int size = l->size * l->size * l->c * l->n * locations;
      fread(l->biases, sizeof(float), l->outputs, fp);
      fread(l->weights, sizeof(float), size, fp);
#ifdef GPU
      if (gpu_index >= 0)
        PushLocalLayer(l);
#endif
    }
    if (feof(fp))
      break;
  }
  fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
  fclose(fp);
}

void LoadWeights(Network* net, char const* filename)
{
  LoadWeightsUpTo(net, filename, net->n);
}

// load network & force - set batch size
Network* LoadNetworkCustom(
    char const* model_file, char const* weights_file, int clear, int batch)
{
  printf(" Try to load model: %s, weights: %s, clear = %d \n", model_file,
      weights_file, clear);

  Network* net = (Network*)xcalloc(1, sizeof(Network));
  ParseNetworkCfgCustom(net, model_file, batch, 1);
  if (weights_file && weights_file[0] != 0)
  {
    printf(" Try to load weights: %s \n", weights_file);
    LoadWeights(net, weights_file);
  }

  FuseConvBatchNorm(net);
  if (clear)
  {
    (*net->seen) = 0;
    (*net->cur_iteration) = 0;
  }

  return net;
}

// load network & get batch size from cfg-file
Network* LoadNetwork(
    char const* model_file, char const* weights_file, int clear)
{
  printf(" Try to load cfg: %s, clear = %d \n", model_file, clear);

  Network* net = (Network*)xcalloc(1, sizeof(Network));
  ParseNetworkCfg(net, model_file);
  if (weights_file && weights_file[0] != 0)
  {
    printf(" Try to load weights: %s \n", weights_file);
    LoadWeights(net, weights_file);
  }

  if (clear)
  {
    (*net->seen) = 0;
    (*net->cur_iteration) = 0;
  }

  return net;
}
