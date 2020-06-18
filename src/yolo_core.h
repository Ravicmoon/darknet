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

struct tree;
typedef struct tree tree;

extern int gpu_index;

// tree.h
typedef struct tree
{
  int* leaf;
  int n;
  int* parent;
  int* child;
  int* group;
  char** name;

  int groups;
  int* group_size;
  int* group_offset;
} tree;

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

// parser.h
typedef enum
{
  NO_WEIGHTS,
  PER_FEATURE,
  PER_CHANNEL
} WEIGHTS_TYPE_T;

// parser.h
typedef enum
{
  NO_NORMALIZATION,
  RELU_NORMALIZATION,
  SOFTMAX_NORMALIZATION
} WEIGHTS_NORMALIZATION_T;

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
  SOFTMAX,
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
  REGION,
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
  L1,
  SEG,
  SMOOTH,
  WGAN
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
  int spatial;
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
  int noloss;
  int softmax;
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
  float mask_scale;
  float class_scale;
  int bias_match;
  float random;
  float ignore_thresh;
  float truth_thresh;
  float iou_thresh;
  float thresh;
  float focus;
  int classfix;
  int absolute;

  int onlyforward;
  int stopbackward;
  int train_only_bn;
  int dont_update;
  int burnin_update;
  int dontload;
  int dontloadscales;

  float temperature;
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
  WEIGHTS_TYPE_T weights_type;
  WEIGHTS_NORMALIZATION_T weights_normalization;
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

  tree* softmax_tree;

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
  int n;
  int batch;
  uint64_t* seen;
  int* cur_iteration;
  float loss_scale;
  int* t;
  float epoch;
  int subdiv;
  layer* layers;
  float* output;
  LearningRatePolicy policy;
  int benchmark_layers;

  float learning_rate;
  float learning_rate_min;
  float learning_rate_max;
  int batches_per_cycle;
  int batches_cycle_mult;
  float momentum;
  float decay;
  float gamma;
  float scale;
  float power;
  int time_steps;
  int step;
  int max_batches;
  float* seq_scales;
  float* scales;
  int* steps;
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
  int attention;
  int adversarial;
  float adversarial_lr;
  int letter_box;
  float angle;
  float aspect;
  float exposure;
  float saturation;
  float hue;
  int random;
  int track;
  int augment_speed;
  int seq_subdiv;
  int init_seq_subdiv;
  int curr_subdiv;
  int try_fix_nan;

  int gpu_index;
  tree* hierarchy;

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

// network.c -batch inference
typedef struct det_num_pair
{
  int num;
  Detection* dets;
} det_num_pair, *pdet_num_pair;

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
  CLASSIFICATION_DATA,
  DETECTION_DATA,
  CAPTCHA_DATA,
  REGION_DATA,
  IMAGE_DATA,
  COMPARE_DATA,
  WRITING_DATA,
  SWAG_DATA,
  TAG_DATA,
  OLD_CLASSIFICATION_DATA,
  STUDY_DATA,
  DET_DATA,
  SUPER_DATA,
  LETTERBOX_DATA,
  REGRESSION_DATA,
  SEGMENTATION_DATA,
  INSTANCE_DATA,
  ISEG_DATA
} data_type;

// data.h
typedef struct load_args
{
  int threads;
  char** paths;
  char* path;
  int n;
  int m;
  char** labels;
  int h;
  int w;
  int c;  // color depth
  int out_w;
  int out_h;
  int nh;
  int nw;
  int num_boxes;
  int min, max, size;
  int classes;
  int scale;
  int center;
  int coords;
  int mini_batch;
  int track;
  int augment_speed;
  int letter_box;
  int show_imgs;
  int dontuse_opencv;
  float jitter;
  int flip;
  int gaussian_noise;
  int blur;
  int mixup;
  float label_smooth_eps;
  float angle;
  float aspect;
  float saturation;
  float exposure;
  float hue;
  data* d;
  Image* im;
  Image* resized;
  data_type type;
  tree* hierarchy;
} load_args;

// data.h
typedef struct BoxLabel
{
  int id;
  float x, y, w, h;
  float left, right, top, bottom;
} BoxLabel;

// parser.c
LIB_API Network* LoadNetwork(
    char const* model_file, char const* weights_file, int clear);
LIB_API Network* LoadNetworkCustom(
    char const* model_file, char const* weights_file, int clear, int batch);
LIB_API void FreeNetwork(Network* net);

// network.h
LIB_API float* NetworkPredict(Network* net, float* input);
LIB_API Detection* GetNetworkBoxes(Network* net, int w, int h, float thresh,
    float hier, int* map, int relative, int* num, int letter);
LIB_API void FreeDetections(Detection* dets, int n);
LIB_API void FuseConvBatchNorm(Network* net);
LIB_API void calculate_binary_weights(Network net);
LIB_API char* Detection2Json(Detection* dets, int nboxes, int classes,
    char** names, long long int frame_id, char const* filename);

LIB_API layer* get_network_layer(Network* net, int i);
LIB_API Detection* MakeNetworkBoxes(Network* net, float thresh, int* num);

LIB_API void TrainDetector(char const* data_file, char const* model_file,
    char const* weights_file, int num_gpus, bool clear, bool show_imgs,
    bool calc_map, int benchmark_layers);
LIB_API float ValidateDetector(Network* net, char const* data_file,
    float const iou_thresh, int letter_box);

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
LIB_API void* cuda_get_context();

// utils.h
LIB_API void free_ptrs(void** ptrs, int n);
LIB_API void top_k(float* a, int n, int k, int* index);

// tree.h
LIB_API tree* read_tree(char* filename);

// gemm.h
LIB_API void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // YOLO_CORE_API
