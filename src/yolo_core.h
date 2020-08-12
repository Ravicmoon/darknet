#ifndef YOLO_CORE_API
#define YOLO_CORE_API

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GPU

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#ifdef CUDNN
#include <cudnn.h>
#endif  // CUDNN
#endif  // GPU

#include "box.h"
#include "image.h"
#include "libapi.h"
#include "option_list.h"
#include "utils.h"

#define SECRET_NUM -1234

typedef enum
{
  UNUSED_DEF_VAL
} UNUSED_ENUM_TYPE;

#ifdef __cplusplus
extern "C" {
#endif

struct Network;
typedef struct Network Network;

struct NetworkState;
typedef struct NetworkState NetworkState;

struct layer;
typedef struct layer layer;

struct Image;
typedef struct Image Image;

struct Detection;
typedef struct Detection Detection;

struct load_args;
typedef struct load_args load_args;

struct data;
typedef struct data data;

// activations.h
typedef enum
{
  LOGISTIC,
  RELU,
  RELU6,
  RELIE,
  LINEAR,
  RAMP,
  TANH,
  PLSE,
  LEAKY,
  ELU,
  LOGGY,
  STAIR,
  HARDTAN,
  LHTAN,
  SELU,
  GELU,
  SWISH,
  MISH,
  NORM_CHAN,
  NORM_CHAN_SOFTMAX,
  NORM_CHAN_SOFTMAX_MAXVAL
} ACTIVATION;

// parser.h
typedef enum
{
  YOLO_CENTER = 1 << 0,
  YOLO_LEFT_TOP = 1 << 1,
  YOLO_RIGHT_BOTTOM = 1 << 2
} YOLO_POINT;

// activations.h
typedef enum
{
  MULT,
  ADD,
  SUB,
  DIV
} BINARY_ACTIVATION;

// layer.h
typedef enum
{
  CONVOLUTIONAL,
  CONNECTED,
  MAXPOOL,
  LOCAL_AVGPOOL,
  DETECTION,
  DROPOUT,
  CROP,
  ROUTE,
  COST,
  AVGPOOL,
  LOCAL,
  SHORTCUT,
  SCALE_CHANNELS,
  ACTIVE,
  BATCHNORM,
  NETWORK,
  XNOR,
  YOLO,
  GAUSSIAN_YOLO,
  REORG,
  REORG_OLD,
  UPSAMPLE,
  EMPTY,
  BLANK
} LAYER_TYPE;

// layer.h
typedef enum
{
  SSE,
  MASKED,
  SMOOTH,
} COST_TYPE;

// layer.h
struct layer
{
  LAYER_TYPE type;
  ACTIVATION activation;
  COST_TYPE cost_type;
  void (*forward)(struct layer*, struct NetworkState);
  void (*backward)(struct layer*, struct NetworkState);
  void (*update)(struct layer*, int, float, float, float);
  void (*forward_gpu)(struct layer*, struct NetworkState);
  void (*backward_gpu)(struct layer*, struct NetworkState);
  void (*update_gpu)(struct layer*, int, float, float, float, float);
  layer* share_layer;
  int train;
  int batch_normalize;
  int shortcut;
  int batch;
  int forced;
  int inputs;
  int outputs;
  int nweights;
  int nbiases;
  int truths;
  int h, w, c;
  int out_h, out_w, out_c;
  int n;
  int max_boxes;
  int groups;
  int group_id;
  int size;
  int side;
  int stride;
  int stride_x;
  int stride_y;
  int dilation;
  int antialiasing;
  int maxpool_depth;
  int out_channels;
  int reverse;
  int flatten;
  int pad;
  int sqrt;
  int flip;
  int index;
  int scale_wh;
  int binary;
  int xnor;
  int use_bin_output;
  int keep_delta_gpu;
  int optimized_memory;
  int steps;

  float angle;
  float jitter;
  float saturation;
  float exposure;
  float shift;
  float ratio;
  float learning_rate_scale;
  float clip;
  int focal_loss;
  float* classes_multipliers;
  float label_smooth_eps;
  int classes;
  int coords;
  int rescore;
  int objectness;
  int does_cost;
  int joint;
  int noadjust;
  int reorg;
  int log;
  int tanh;
  int* mask;
  int total;
  float bflops;

  int adam;
  float B1;
  float B2;
  float eps;

  int t;

  float alpha;
  float beta;
  float kappa;

  float coord_scale;
  float object_scale;
  float noobject_scale;
  float class_scale;
  float random;
  float ignore_thresh;
  float truth_thresh;
  float iou_thresh;
  float thresh;
  float focus;

  int onlyforward;
  int stopbackward;
  int train_only_bn;
  int dont_update;
  int burnin_update;
  int dontload;
  int dontloadscales;

  float probability;
  float dropblock_size_rel;
  int dropblock_size_abs;
  int dropblock;
  float scale;

  int receptive_w;
  int receptive_h;
  int receptive_w_scale;
  int receptive_h_scale;

  int* indexes;
  int* input_layers;
  int* input_sizes;
  float** layers_output;
  float** layers_delta;
  int* map;
  int* counts;
  float** sums;
  float* rand;
  float* cost;

  float* binary_weights;

  float* biases;
  float* bias_updates;

  float* scales;
  float* scale_updates;

  float* weights;
  float* weight_updates;

  float scale_x_y;
  float max_delta;
  float uc_normalizer;
  float iou_normalizer;
  float cls_normalizer;
  IOU_LOSS iou_loss;
  IOU_LOSS iou_thresh_kind;
  NMS_KIND nms_kind;
  float beta_nms;
  YOLO_POINT yolo_point;

  char* align_bit_weights_gpu;
  float* mean_arr_gpu;
  float* align_workspace_gpu;
  float* transposed_align_workspace_gpu;
  int align_workspace_size;

  char* align_bit_weights;
  float* mean_arr;
  int align_bit_weights_size;
  int lda_align;
  int new_lda;
  int bit_align;

  float* col_image;
  float* delta;
  float* output;
  float* activation_input;
  int delta_pinned;
  int output_pinned;
  float* loss;
  float* squared;
  float* norms;

  float* mean;
  float* variance;

  float* mean_delta;
  float* variance_delta;

  float* rolling_mean;
  float* rolling_variance;

  float* x;
  float* x_norm;

  float* m;
  float* v;

  float* bias_m;
  float* bias_v;
  float* scale_m;
  float* scale_v;

  float* binary_input;
  uint32_t* bin_re_packed_input;
  char* t_bit_input;

  struct layer* input_layer;

  size_t workspace_size;

  int* indexes_gpu;

  // adam
  float* m_gpu;
  float* v_gpu;
  float* bias_m_gpu;
  float* scale_m_gpu;
  float* bias_v_gpu;
  float* scale_v_gpu;

  float* binary_input_gpu;
  float* binary_weights_gpu;
  float* bin_conv_shortcut_in_gpu;
  float* bin_conv_shortcut_out_gpu;

  float* mean_gpu;
  float* variance_gpu;
  float* m_cbn_avg_gpu;
  float* v_cbn_avg_gpu;

  float* rolling_mean_gpu;
  float* rolling_variance_gpu;

  float* variance_delta_gpu;
  float* mean_delta_gpu;

  float* col_image_gpu;

  float* x_gpu;
  float* x_norm_gpu;
  float* weights_gpu;
  float* weight_updates_gpu;
  float* weight_change_gpu;

  float* weights_gpu16;
  float* weight_updates_gpu16;

  float* biases_gpu;
  float* bias_updates_gpu;
  float* bias_change_gpu;

  float* scales_gpu;
  float* scale_updates_gpu;
  float* scale_change_gpu;

  float* input_antialiasing_gpu;
  float* output_gpu;
  float* activation_input_gpu;
  float* loss_gpu;
  float* delta_gpu;
  float* rand_gpu;
  float* drop_blocks_scale;
  float* drop_blocks_scale_gpu;
  float* squared_gpu;
  float* norms_gpu;

  int* input_sizes_gpu;
  float** layers_output_gpu;
  float** layers_delta_gpu;
#ifdef CUDNN
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
  cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
  cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
  cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
  cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc,
      normDstTensorDescF16;
  cudnnFilterDescriptor_t weightDesc, weightDesc16;
  cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
  cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
  cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
  cudnnPoolingDescriptor_t poolingDesc;
#else   // CUDNN
  void *srcTensorDesc, *dstTensorDesc;
  void *srcTensorDesc16, *dstTensorDesc16;
  void *dsrcTensorDesc, *ddstTensorDesc;
  void *dsrcTensorDesc16, *ddstTensorDesc16;
  void *normTensorDesc, *normDstTensorDesc, *normDstTensorDescF16;
  void *weightDesc, *weightDesc16;
  void *dweightDesc, *dweightDesc16;
  void* convDesc;
  UNUSED_ENUM_TYPE fw_algo, fw_algo16;
  UNUSED_ENUM_TYPE bd_algo, bd_algo16;
  UNUSED_ENUM_TYPE bf_algo, bf_algo16;
  void* poolingDesc;
#endif  // CUDNN
};

// network.h
typedef enum
{
  CONSTANT,
  STEP,
  EXP,
  POLY,
  STEPS,
  SIG,
  RANDOM,
  SGDR
} LearningRatePolicy;

// network.h
typedef struct Network
{
  int max_epoch;
  int max_iter;

  int n;
  int batch;
  int subdiv;
  uint64_t seen;
  int curr_iter;
  float loss_scale;
  int* t;
  layer* layers;
  float* output;
  LearningRatePolicy policy;
  int benchmark_layers;

  float lr;
  float lr_min;
  int sgdr_cycle;
  int sgdr_mult;
  float momentum;
  float decay;
  float gamma;
  float scale;
  float power;
  int step;
  float* steps;
  float* scales;
  int num_steps;
  int burn_in;
  int cudnn_half;

  int adam;
  float B1;
  float B2;
  float eps;

  int inputs;
  int outputs;
  int truths;
  int notruth;
  int h, w, c;
  int max_crop;
  int min_crop;
  float max_ratio;
  float min_ratio;
  int center;
  int flip;  // horizontal flip 50% probability augmentaiont for classifier
             // training (default = 1)
  int gaussian_noise;
  int blur;
  int mixup;
  float label_smooth_eps;
  int resize_step;
  float angle;
  float aspect;
  float exposure;
  float saturation;
  float hue;
  int curr_subdiv;

  int gpu_index;

  float* input;
  float* truth;
  float* delta;
  float* workspace;
  int train;
  int index;
  float* cost;

  float* delta_gpu;
  float* output_gpu;

  float* input_state_gpu;
  float* input_pinned_cpu;
  int input_pinned_cpu_flag;

  float** input_gpu;
  float** truth_gpu;
  float** input16_gpu;
  float** output16_gpu;
  size_t* max_input16_size;
  size_t* max_output16_size;
  int wait_stream;

  float* global_delta_gpu;
  float* state_delta_gpu;
  size_t max_delta_gpu_size;

  int optimized_memory;
  size_t workspace_size_limit;
} Network;

// network.h
typedef struct NetworkState
{
  float* truth;
  float* input;
  float* delta;
  float* workspace;
  int train;
  int index;
  Network* net;
} NetworkState;

// matrix.h
typedef struct matrix
{
  int rows, cols;
  float** vals;
} matrix;

// data.h
typedef struct data
{
  int w, h;
  matrix X;
  matrix y;
  int shallow;
  int* num_boxes;
  Box** boxes;
} data;

// data.h
typedef enum
{
  DETECTION_DATA,
  IMAGE_DATA,
} data_type;

// data.h
typedef struct load_args
{
  int threads;
  char** paths;
  char const* path;
  int n;
  int m;
  int h;
  int w;
  int c;  // color depth
  int num_boxes;
  int min, max, size;
  int classes;
  int scale;
  int show_imgs;
  float jitter;
  int flip;
  int gaussian_noise;
  int blur;
  int mixup;
  float aspect;
  float saturation;
  float exposure;
  float hue;
  data* d;
  Image* im;
  Image* resized;
  data_type type;
} load_args;

// data.h
typedef struct BoxLabel
{
  int id;
  float x, y, w, h;
  float left, right, top, bottom;
} BoxLabel;

// parser.c
LIB_API void LoadNetwork(Network* net, char const* model_file,
    char const* weights_file, bool train = false, bool clear = false);
LIB_API void FreeNetwork(Network* net);

// network.h
LIB_API float* NetworkPredict(Network* net, float* input);
LIB_API Detection* GetNetworkBoxes(Network* net, float thresh, int* num);
LIB_API void FreeDetections(Detection* dets, int n);
LIB_API void FuseConvBatchNorm(Network* net);
LIB_API void calculate_binary_weights(Network net);
LIB_API char* Detection2Json(Detection* dets, int nboxes, int classes,
    char** names, long long int frame_id, char const* filename);

LIB_API Detection* MakeNetworkBoxes(Network* net, float thresh, int* num);

LIB_API void TrainDetector(Metadata const& md, std::string model_file,
    std::string weights_file, int num_gpus, bool clear, bool show_imgs,
    bool calc_map, int benchmark_layers);
LIB_API float ValidateDetector(
    Metadata const& md, Network* net, float const iou_thresh);

// layer.h
LIB_API void free_layer(layer* l, bool keep_cudnn_desc = false);

// data.c
LIB_API void free_data(data d);
LIB_API pthread_t load_data(load_args args);
LIB_API void free_load_threads(void* ptr);
LIB_API pthread_t load_data_in_thread(load_args args);
LIB_API void* load_thread(void* ptr);

// dark_cuda.h
LIB_API void cuda_pull_array(float* x_gpu, float* x, size_t n);
LIB_API void cuda_pull_array_async(float* x_gpu, float* x, size_t n);
LIB_API void cuda_set_device(int n);

// gemm.h
LIB_API void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // YOLO_CORE_API
