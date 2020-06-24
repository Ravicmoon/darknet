#include "network.h"

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

#include "activation_layer.h"
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
#include "local_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "reorg_old_layer.h"
#include "route_layer.h"
#include "scale_channels_layer.h"
#include "shortcut_layer.h"
#include "upsample_layer.h"
#include "utils.h"
#include "yolo_layer.h"

int64_t GetCurrentIteration(Network* net) { return *net->cur_iteration; }

int GetCurrentBatch(Network* net)
{
  return (*net->seen) / (net->batch * net->subdiv);
}

float GetCurrentRate(Network* net)
{
  int batch_num = GetCurrentBatch(net);
  int i;
  float rate;
  if (batch_num < net->burn_in)
    return net->learning_rate *
           pow((float)batch_num / net->burn_in, net->power);
  switch (net->policy)
  {
    case CONSTANT:
      return net->learning_rate;
    case STEP:
      return net->learning_rate * pow(net->scale, batch_num / net->step);
    case STEPS:
      rate = net->learning_rate;
      for (i = 0; i < net->num_steps; ++i)
      {
        if (net->steps[i] > batch_num)
          return rate;
        rate *= net->scales[i];
      }
      return rate;
    case EXP:
      return net->learning_rate * pow(net->gamma, batch_num);
    case POLY:
      return net->learning_rate *
             pow(1 - (float)batch_num / net->max_batches, net->power);
    case RANDOM:
      return net->learning_rate * pow(RandUniform(0, 1), net->power);
    case SIG:
      return net->learning_rate *
             (1. / (1. + exp(net->gamma * (batch_num - net->step))));
    case SGDR: {
      int last_iteration_start = 0;
      int cycle_size = net->batches_per_cycle;
      while ((last_iteration_start + cycle_size) < batch_num)
      {
        last_iteration_start += cycle_size;
        cycle_size *= net->batches_cycle_mult;
      }
      rate = net->learning_rate_min +
             0.5 * (net->learning_rate - net->learning_rate_min) *
                 (1. + cos((float)(batch_num - last_iteration_start) *
                           3.14159265 / cycle_size));

      return rate;
    }
    default:
      fprintf(stderr, "Policy is weird!\n");
      return net->learning_rate;
  }
}

void AllocateNetwork(Network* net, int n)
{
  net->n = n;
  net->layers = (layer*)xcalloc(net->n, sizeof(layer));
  net->seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));
  net->cur_iteration = (int*)xcalloc(1, sizeof(int));
#ifdef GPU
  net->input_gpu = (float**)xcalloc(1, sizeof(float*));
  net->truth_gpu = (float**)xcalloc(1, sizeof(float*));

  net->input16_gpu = (float**)xcalloc(1, sizeof(float*));
  net->output16_gpu = (float**)xcalloc(1, sizeof(float*));
  net->max_input16_size = (size_t*)xcalloc(1, sizeof(size_t));
  net->max_output16_size = (size_t*)xcalloc(1, sizeof(size_t));
#endif
}

void ForwardNetwork(Network* net, NetworkState state)
{
  state.workspace = net->workspace;
  for (int i = 0; i < net->n; ++i)
  {
    state.index = i;
    layer* l = &net->layers[i];
    if (l->delta && state.train)
      scal_cpu(l->outputs * l->batch, 0, l->delta, 1);

    l->forward(l, state);
    state.input = l->output;
  }
}

void UpdateNetwork(Network* net)
{
  int actual_batch = net->batch * net->subdiv;

  float rate = GetCurrentRate(net);
  for (int i = 0; i < net->n; ++i)
  {
    layer* l = &net->layers[i];
    if (l->update)
      l->update(l, actual_batch, rate, net->momentum, net->decay);
  }
}

float* GetNetworkOutput(Network* net)
{
#ifdef GPU
  if (gpu_index >= 0)
    return GetNetworkOutputGpu(net);
#endif
  int i;
  for (i = net->n - 1; i > 0; --i)
  {
    if (net->layers[i].type != COST)
      break;
  }

  return net->layers[i].output;
}

float GetNetworkCost(Network* net)
{
  float sum = 0;
  int count = 0;
  for (int i = 0; i < net->n; ++i)
  {
    if (net->layers[i].cost)
    {
      sum += net->layers[i].cost[0];
      ++count;
    }
  }
  return sum / count;
}

void BackwardNetwork(Network* net, NetworkState state)
{
  float* original_input = state.input;
  float* original_delta = state.delta;
  state.workspace = net->workspace;
  for (int i = net->n - 1; i >= 0; --i)
  {
    state.index = i;
    if (i == 0)
    {
      state.input = original_input;
      state.delta = original_delta;
    }
    else
    {
      layer* prev = &net->layers[i - 1];
      state.input = prev->output;
      state.delta = prev->delta;
    }
    layer* l = &net->layers[i];
    if (l->stopbackward)
      break;
    if (l->onlyforward)
      continue;
    l->backward(l, state);
  }
}

float TrainNetworkDatum(Network* net, float* x, float* y)
{
#ifdef GPU
  if (gpu_index >= 0)
    return TrainNetworkDatumGpu(net, x, y);
#endif
  NetworkState state = {0};
  *net->seen += net->batch;
  state.index = 0;
  state.net = net;
  state.input = x;
  state.delta = 0;
  state.truth = y;
  state.train = 1;

  ForwardNetwork(net, state);
  BackwardNetwork(net, state);

  return GetNetworkCost(net);
}

float TrainNetwork(Network* net, data d)
{
  assert(d.X.rows % net->batch == 0);

  int batch = net->batch;
  int n = d.X.rows / batch;
  float* X = (float*)xcalloc(batch * d.X.cols, sizeof(float));
  float* y = (float*)xcalloc(batch * d.y.cols, sizeof(float));

  float sum = 0;
  for (int i = 0; i < n; ++i)
  {
    get_next_batch(d, batch, i * batch, X, y);
    net->curr_subdiv = i;
    sum += TrainNetworkDatum(net, X, y);
  }
  (*net->cur_iteration) += 1;

#ifdef GPU
  UpdateNetworkGpu(net);
#else   // GPU
  UpdateNetwork(net);
#endif  // GPU

  free(X);
  free(y);

  return (float)sum / (n * batch);
}

int GetNetworkInputSize(Network* net) { return net->layers[0].inputs; }

int GetNetworkOutputSize(Network* net)
{
  int i;
  for (i = net->n - 1; i > 0; --i)
  {
    if (net->layers[i].type != COST)
      break;
  }

  return net->layers[i].outputs;
}

void ResizeNetwork(Network* net, int w, int h)
{
#ifdef GPU
  cuda_set_device(net->gpu_index);
  if (gpu_index >= 0)
  {
    cuda_free(net->workspace);
    if (net->input_gpu)
    {
      cuda_free(*net->input_gpu);
      *net->input_gpu = 0;
      cuda_free(*net->truth_gpu);
      *net->truth_gpu = 0;
    }

    if (net->input_state_gpu)
      cuda_free(net->input_state_gpu);
    if (net->input_pinned_cpu)
    {
      if (net->input_pinned_cpu_flag)
        cudaFreeHost(net->input_pinned_cpu);
      else
        free(net->input_pinned_cpu);
    }
  }
#endif

  net->w = w;
  net->h = h;
  int inputs = 0;
  size_t workspace_size = 0;

  for (int i = 0; i < net->n; ++i)
  {
    layer* l = &net->layers[i];

    if (l->type == CONVOLUTIONAL)
    {
      resize_convolutional_layer(l, w, h);
    }
    else if (l->type == CROP)
    {
      ResizeCropLayer(l, w, h);
    }
    else if (l->type == MAXPOOL)
    {
      ResizeMaxpoolLayer(l, w, h);
    }
    else if (l->type == LOCAL_AVGPOOL)
    {
      ResizeMaxpoolLayer(l, w, h);
    }
    else if (l->type == BATCHNORM)
    {
      ResizeBatchnormLayer(l, w, h);
    }
    else if (l->type == YOLO)
    {
      ResizeYoloLayer(l, w, h);
    }
    else if (l->type == GAUSSIAN_YOLO)
    {
      ResizeGaussianYoloLayer(l, w, h);
    }
    else if (l->type == ROUTE)
    {
      ResizeRouteLayer(l, net);
    }
    else if (l->type == SHORTCUT)
    {
      ResizeShortcutLayer(l, w, h, net);
    }
    else if (l->type == SCALE_CHANNELS)
    {
      ResizeScaleChannelsLayer(l, net);
    }
    else if (l->type == DROPOUT)
    {
      ResizeDropoutLayer(l, inputs);
      l->out_w = l->w = w;
      l->out_h = l->h = h;
      l->output = net->layers[i - 1].output;
      l->delta = net->layers[i - 1].delta;
#ifdef GPU
      l->output_gpu = net->layers[i - 1].output_gpu;
      l->delta_gpu = net->layers[i - 1].delta_gpu;
#endif
    }
    else if (l->type == UPSAMPLE)
    {
      ResizeUpsampleLayer(l, w, h);
    }
    else if (l->type == REORG)
    {
      resize_reorg_layer(l, w, h);
    }
    else if (l->type == REORG_OLD)
    {
      ResizeReorgOldLayer(l, w, h);
    }
    else if (l->type == AVGPOOL)
    {
      ResizeAvgpoolLayer(l, w, h);
    }
    else if (l->type == COST)
    {
      ResizeCostLayer(l, inputs);
    }
    else
    {
      fprintf(stderr, "Resizing type %d \n", (int)l->type);
      error("Cannot resize this type of layer");
    }

    if (l->workspace_size > workspace_size)
      workspace_size = l->workspace_size;

    inputs = l->outputs;
    w = l->out_w;
    h = l->out_h;
  }
#ifdef GPU
  const int size = GetNetworkInputSize(net) * net->batch;
  if (gpu_index >= 0)
  {
    printf(" try to allocate additional workspace_size = %1.2f MB \n",
        (float)workspace_size / 1000000);
    net->workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
    net->input_state_gpu = cuda_make_array(0, size);
    if (cudaSuccess == cudaHostAlloc(&net->input_pinned_cpu,
                           size * sizeof(float), cudaHostRegisterMapped))
      net->input_pinned_cpu_flag = 1;
    else
    {
      cudaGetLastError();  // reset CUDA-error
      net->input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
      net->input_pinned_cpu_flag = 0;
    }
    printf(" CUDA allocate done! \n");
  }
  else
  {
    free(net->workspace);
    net->workspace = (float*)xcalloc(1, workspace_size);
    if (!net->input_pinned_cpu_flag)
      net->input_pinned_cpu =
          (float*)xrealloc(net->input_pinned_cpu, size * sizeof(float));
  }
#else
  free(net->workspace);
  net->workspace = (float*)xcalloc(1, workspace_size);
#endif
}

float* NetworkPredict(Network* net, float* input)
{
#ifdef GPU
  if (gpu_index >= 0)
    return NetworkPredictGpu(net, input);
#endif

  NetworkState state = {0};
  state.net = net;
  state.index = 0;
  state.input = input;
  state.truth = 0;
  state.train = 0;
  state.delta = 0;

  ForwardNetwork(net, state);

  return GetNetworkOutput(net);
}

int NumDetections(Network* net, float thresh)
{
  int s = 0;
  for (int i = 0; i < net->n; ++i)
  {
    layer const* l = &net->layers[i];
    if (l->type == YOLO)
      s += YoloNumDetections(l, thresh);

    if (l->type == GAUSSIAN_YOLO)
      s += GaussianYoloNumDetections(l, thresh);

    if (l->type == DETECTION)
      s += l->w * l->h * l->n;
  }
  return s;
}

Detection* MakeNetworkBoxes(Network* net, float thresh, int* num)
{
  layer* l = &net->layers[net->n - 1];

  int num_boxes = NumDetections(net, thresh);
  if (num != NULL)
    *num = num_boxes;

  Detection* dets = (Detection*)xcalloc(num_boxes, sizeof(Detection));
  for (int i = 0; i < num_boxes; ++i)
  {
    dets[i].prob = (float*)xcalloc(l->classes, sizeof(float));

    if (l->type == GAUSSIAN_YOLO)
      dets[i].uc = (float*)xcalloc(4, sizeof(float));

    if (l->coords > 4)
      dets[i].mask = (float*)xcalloc(l->coords - 4, sizeof(float));
  }

  return dets;
}

void FillNetworkBoxes(Network* net, int w, int h, float thresh, float hier,
    int* map, int relative, Detection* dets, int letter)
{
  int prev_classes = -1;
  for (int i = 0; i < net->n; ++i)
  {
    layer* l = &net->layers[i];
    if (l->type == YOLO)
    {
      int count = GetYoloDetections(
          l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
      dets += count;
      if (prev_classes < 0)
        prev_classes = l->classes;
      else if (prev_classes != l->classes)
      {
        printf(
            " Error: Different [yolo] layers have different number of classes "
            "= %d and %d - check your cfg-file! \n",
            prev_classes, l->classes);
      }
    }

    if (l->type == GAUSSIAN_YOLO)
    {
      int count = GetGaussianYoloDetections(
          l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
      dets += count;
    }

    if (l->type == DETECTION)
    {
      GetDetectionDetections(l, w, h, thresh, dets);
      dets += l->w * l->h * l->n;
    }
  }
}

Detection* GetNetworkBoxes(Network* net, int w, int h, float thresh, float hier,
    int* map, int relative, int* num, int letter)
{
  Detection* dets = MakeNetworkBoxes(net, thresh, num);
  FillNetworkBoxes(net, w, h, thresh, hier, map, relative, dets, letter);
  return dets;
}

void FreeDetections(Detection* dets, int n)
{
  for (int i = 0; i < n; ++i)
  {
    free(dets[i].prob);
    if (dets[i].uc)
      free(dets[i].uc);
    if (dets[i].mask)
      free(dets[i].mask);
  }
  free(dets);
}

// JSON format:
//{
// "frame_id":8990,
// "objects":[
//  {"class_id":4, "name":"aeroplane", "relative
//  coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455,
//  "height":0.020396}, "confidence":0.793070},
//  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831,
//  "center_y":0.630203, "width":0.057455, "height":0.020396},
//  "confidence":0.265497}
// ]
//},

char* Detection2Json(Detection* dets, int nboxes, int classes, char** names,
    long long int frame_id, char const* filename)
{
  const float thresh = 0.005;  // function get_network_boxes() has already
                               // filtred dets by actual threshold

  char* send_buf = (char*)calloc(1024, sizeof(char));
  if (!send_buf)
    return 0;

  if (filename)
  {
    sprintf(send_buf,
        "{\n \"frame_id\":%lld, \n \"filename\":\"%s\", \n \"objects\": [ \n",
        frame_id, filename);
  }
  else
  {
    sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"objects\": [ \n", frame_id);
  }

  int class_id = -1;
  for (int i = 0; i < nboxes; ++i)
  {
    for (int j = 0; j < classes; ++j)
    {
      int show = strncmp(names[j], "dont_show", 9);
      if (dets[i].prob[j] > thresh && show)
      {
        if (class_id != -1)
          strcat(send_buf, ", \n");

        class_id = j;
        char* buf = (char*)calloc(2048, sizeof(char));
        if (!buf)
          return 0;

        sprintf(buf,
            "  {\"class_id\":%d, \"name\":\"%s\", "
            "\"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, "
            "\"width\":%f, \"height\":%f}, \"confidence\":%f}",
            j, names[j], dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w,
            dets[i].bbox.h, dets[i].prob[j]);

        int send_buf_len = strlen(send_buf);
        int buf_len = strlen(buf);
        int total_len = send_buf_len + buf_len + 100;
        send_buf = (char*)realloc(send_buf, total_len * sizeof(char));
        if (!send_buf)
        {
          if (buf)
            free(buf);
          return 0;
        }
        strcat(send_buf, buf);
        free(buf);
      }
    }
  }
  strcat(send_buf, "\n ] \n}");

  return send_buf;
}

void FreeNetwork(Network* net)
{
  for (int i = 0; i < net->n; ++i)
  {
    free_layer(&net->layers[i]);
  }
  free(net->layers);

  free(net->seq_scales);
  free(net->scales);
  free(net->steps);
  free(net->seen);
  free(net->cur_iteration);

#ifdef GPU
  if (gpu_index >= 0)
    cuda_free(net->workspace);
  else
    free(net->workspace);
  free_pinned_memory();
  if (net->input_state_gpu)
    cuda_free(net->input_state_gpu);
  if (net->input_pinned_cpu)
  {  // CPU
    if (net->input_pinned_cpu_flag)
      cudaFreeHost(net->input_pinned_cpu);
    else
      free(net->input_pinned_cpu);
  }
  if (*net->input_gpu)
    cuda_free(*net->input_gpu);
  if (*net->truth_gpu)
    cuda_free(*net->truth_gpu);
  if (net->input_gpu)
    free(net->input_gpu);
  if (net->truth_gpu)
    free(net->truth_gpu);

  if (*net->input16_gpu)
    cuda_free(*net->input16_gpu);
  if (*net->output16_gpu)
    cuda_free(*net->output16_gpu);
  if (net->input16_gpu)
    free(net->input16_gpu);
  if (net->output16_gpu)
    free(net->output16_gpu);
  if (net->max_input16_size)
    free(net->max_input16_size);
  if (net->max_output16_size)
    free(net->max_output16_size);
#else
  free(net->workspace);
#endif
}

void FuseConvBatchNorm(Network* net)
{
  for (int j = 0; j < net->n; ++j)
  {
    layer* l = &net->layers[j];

    if (l->type == CONVOLUTIONAL)
    {
      if (l->share_layer != NULL)
        l->batch_normalize = 0;

      if (l->batch_normalize)
      {
        for (int f = 0; f < l->n; ++f)
        {
          float std = sqrt(l->rolling_variance[f] + 0.00001f);

          l->biases[f] -= l->scales[f] * l->rolling_mean[f] / std;

          int const filter_size = l->size * l->size * l->c / l->groups;
          for (int i = 0; i < filter_size; ++i)
          {
            l->weights[f * filter_size + i] *= l->scales[f] / std;
          }
        }

        FreeConvBatchnorm(l);
        l->batch_normalize = 0;
#ifdef GPU
        if (gpu_index >= 0)
          PushConvolutionalLayer(l);
#endif
      }
    }
  }
}

void ForwardBlankLayer(layer* l, NetworkState state) {}

void calculate_binary_weights(Network net)
{
  int j;
  for (j = 0; j < net.n; ++j)
  {
    layer* l = &net.layers[j];

    if (l->type == CONVOLUTIONAL)
    {
      // printf(" Merges Convolutional-%d and batch_norm \n", j);

      if (l->xnor)
      {
        // printf("\n %d \n", j);
        // l->lda_align = 256; // 256bit for AVX2    // set in
        // make_convolutional_layer() if (l->size*l->size*l->c >= 2048)
        // l->lda_align = 512;

        binary_align_weights(l);

        if (net.layers[j].use_bin_output)
        {
          l->activation = LINEAR;
        }

#ifdef GPU
        // fuse conv_xnor + shortcut -> conv_xnor
        if ((j + 1) < net.n && net.layers[j].type == CONVOLUTIONAL)
        {
          layer* sc = &net.layers[j + 1];
          if (sc->type == SHORTCUT && sc->w == sc->out_w &&
              sc->h == sc->out_h && sc->c == sc->out_c)
          {
            l->bin_conv_shortcut_in_gpu =
                net.layers[net.layers[j + 1].index].output_gpu;
            l->bin_conv_shortcut_out_gpu = net.layers[j + 1].output_gpu;

            net.layers[j + 1].type = BLANK;
            net.layers[j + 1].forward_gpu = ForwardBlankLayer;
          }
        }
#endif  // GPU
      }
    }
  }
  // printf("\n calculate_binary_weights Done! \n");
}

void copy_cudnn_descriptors(layer src, layer* dst)
{
#ifdef CUDNN
  dst->normTensorDesc = src.normTensorDesc;
  dst->normDstTensorDesc = src.normDstTensorDesc;
  dst->normDstTensorDescF16 = src.normDstTensorDescF16;

  dst->srcTensorDesc = src.srcTensorDesc;
  dst->dstTensorDesc = src.dstTensorDesc;

  dst->srcTensorDesc16 = src.srcTensorDesc16;
  dst->dstTensorDesc16 = src.dstTensorDesc16;
#endif  // CUDNN
}

void CopyNetWeights(Network* net_from, Network* net_to)
{
  for (int k = 0; k < net_from->n; ++k)
  {
    layer* l = &net_from->layers[k];
    layer tmp_layer;
    copy_cudnn_descriptors(net_to->layers[k], &tmp_layer);
    net_to->layers[k] = net_from->layers[k];
    copy_cudnn_descriptors(tmp_layer, &net_to->layers[k]);

    if (l->input_layer)  // for AntiAliasing
    {
      layer tmp_input_layer;
      copy_cudnn_descriptors(*net_to->layers[k].input_layer, &tmp_input_layer);
      net_to->layers[k].input_layer = net_from->layers[k].input_layer;
      copy_cudnn_descriptors(tmp_input_layer, net_to->layers[k].input_layer);
    }

    net_to->layers[k].batch = 1;
    net_to->layers[k].steps = 1;
  }
}
