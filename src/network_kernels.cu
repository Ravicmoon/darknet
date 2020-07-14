#include <assert.h>
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
#include "dark_cuda.h"
#include "data.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "image.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "network.h"
#include "parser.h"
#include "reorg_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "utils.h"

float* GetNetworkOutputGpu(Network* net);

typedef struct time_benchmark_layers
{
  float time;
  int layer_id, layer_type;
} time_benchmark_layers;

int time_comparator(const void* pa, const void* pb)
{
  time_benchmark_layers a = *(time_benchmark_layers*)pa;
  time_benchmark_layers b = *(time_benchmark_layers*)pb;
  float diff = a.time - b.time;
  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  return 0;
}

void ForwardNetworkGpu(Network* net, NetworkState state)
{
  static time_benchmark_layers* avg_time_per_layer = NULL;
  static time_benchmark_layers* sorted_avg_time_per_layer = NULL;
  double start_time = 0.0, end_time = 0.0;
  if (net->benchmark_layers)
  {
    if (!avg_time_per_layer)
    {
      avg_time_per_layer =
          (time_benchmark_layers*)calloc(net->n, sizeof(time_benchmark_layers));
      sorted_avg_time_per_layer =
          (time_benchmark_layers*)calloc(net->n, sizeof(time_benchmark_layers));
    }
    cudaDeviceSynchronize();
  }

  state.workspace = net->workspace;
  for (int i = 0; i < net->n; ++i)
  {
    state.index = i;
    layer* l = &net->layers[i];
    if (l->delta_gpu && state.train)
      fill_ongpu(l->outputs * l->batch, 0, l->delta_gpu, 1);

    if (net->benchmark_layers)
      start_time = GetTimePoint();

    l->forward_gpu(l, state);

    if (net->benchmark_layers)
    {
      CHECK_CUDA(cudaDeviceSynchronize());
      end_time = GetTimePoint();
      double const took_time = (end_time - start_time) / 1000;
      double const alpha = 0.9;
      if (avg_time_per_layer[i].time == 0)
      {
        avg_time_per_layer[i].layer_id = i;
        avg_time_per_layer[i].layer_type = l->type;
        avg_time_per_layer[i].time = took_time;
      }
      else
        avg_time_per_layer[i].time =
            avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

      sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
      printf("\n fw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i,
          l->type, took_time, avg_time_per_layer[i].time);
    }

    if (net->wait_stream)
      cudaStreamSynchronize(get_cuda_stream());
    state.input = l->output_gpu;
  }

  if (net->benchmark_layers)
  {
    printf("\n\nSorted by time (forward):\n");
    qsort(sorted_avg_time_per_layer, net->n, sizeof(time_benchmark_layers),
        time_comparator);
    for (int i = 0; i < net->n; ++i)
    {
      printf("%d - fw-sort-layer %d - type: %d - avg_time %lf ms \n", i,
          sorted_avg_time_per_layer[i].layer_id,
          sorted_avg_time_per_layer[i].layer_type,
          sorted_avg_time_per_layer[i].time);
    }
  }
}

void BackwardNetworkGpu(Network* net, NetworkState state)
{
  static time_benchmark_layers* avg_time_per_layer = NULL;
  static time_benchmark_layers* sorted_avg_time_per_layer = NULL;
  double start_time, end_time;
  if (net->benchmark_layers)
  {
    if (!avg_time_per_layer)
    {
      avg_time_per_layer =
          (time_benchmark_layers*)calloc(net->n, sizeof(time_benchmark_layers));
      sorted_avg_time_per_layer =
          (time_benchmark_layers*)calloc(net->n, sizeof(time_benchmark_layers));
    }
    cudaDeviceSynchronize();
  }

  state.workspace = net->workspace;
  float* original_input = state.input;
  float* original_delta = state.delta;
  for (int i = net->n - 1; i >= 0; --i)
  {
    state.index = i;
    layer* l = &net->layers[i];
    if (l->stopbackward == 1)
      break;
    if (l->stopbackward > GetCurrIter(net))
      break;
    if (i == 0)
    {
      state.input = original_input;
      state.delta = original_delta;
    }
    else
    {
      layer prev = net->layers[i - 1];
      state.input = prev.output_gpu;
      state.delta = prev.delta_gpu;
      if (net->optimized_memory && !prev.keep_delta_gpu)
      {
        state.delta = net->state_delta_gpu;
      }
    }
    if (l->onlyforward)
      continue;

    if (net->benchmark_layers)
    {
      start_time = GetTimePoint();
    }

    l->backward_gpu(l, state);

    if (net->benchmark_layers)
    {
      CHECK_CUDA(cudaDeviceSynchronize());
      end_time = GetTimePoint();
      const double took_time = (end_time - start_time) / 1000;
      const double alpha = 0.9;
      if (avg_time_per_layer[i].time == 0)
      {
        avg_time_per_layer[i].layer_id = i;
        avg_time_per_layer[i].layer_type = l->type;
        avg_time_per_layer[i].time = took_time;
      }
      else
        avg_time_per_layer[i].time =
            avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

      sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
      printf("\n bw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i,
          l->type, took_time, avg_time_per_layer[i].time);
    }

    if (i != 0)
    {
      layer prev = net->layers[i - 1];
      if (net->optimized_memory && state.delta && !prev.keep_delta_gpu)
      {
        if (prev.delta_gpu != state.delta)
          simple_copy_ongpu(
              prev.outputs * prev.batch, state.delta, prev.delta_gpu);
        fill_ongpu(prev.outputs * prev.batch, 0, net->state_delta_gpu, 1);
      }
    }
  }

  if (net->benchmark_layers)
  {
    printf("\n\nSorted by time (backward):\n");
    qsort(sorted_avg_time_per_layer, net->n, sizeof(time_benchmark_layers),
        time_comparator);
    for (int i = 0; i < net->n; ++i)
    {
      printf("%d - bw-sort-layer %d - type: %d - avg_time %lf ms \n", i,
          sorted_avg_time_per_layer[i].layer_id,
          sorted_avg_time_per_layer[i].layer_type,
          sorted_avg_time_per_layer[i].time);
    }
  }
}

void UpdateNetworkGpu(Network* net)
{
  cuda_set_device(net->gpu_index);

  int const actual_batch = net->batch * net->subdiv;
  int const iter = GetCurrIter(net);
  float const lr = GetCurrLr(net);

  for (int i = 0; i < net->n; ++i)
  {
    layer* l = &net->layers[i];
    l->t = iter;

    if (l->burnin_update && (l->burnin_update * net->burn_in > iter))
      continue;

    if (l->train_only_bn)
      continue;

    if (l->update_gpu && l->dont_update < iter)
    {
      l->update_gpu(
          l, actual_batch, lr, net->momentum, net->decay, net->loss_scale);
    }
  }
}

void ForwardBackwardNetworkGpu(Network* net, float* x, float* y)
{
  NetworkState state;
  state.index = 0;
  state.net = net;
  int x_size = GetNetworkInputSize(net) * net->batch;
  int y_size = GetNetworkOutputSize(net) * net->batch;
  if (net->layers[net->n - 1].truths)
    y_size = net->layers[net->n - 1].truths * net->batch;
  if (!*net->input_gpu)
  {
    *net->input_gpu = cuda_make_array(x, x_size);
    *net->truth_gpu = cuda_make_array(y, y_size);
  }
  else
  {
    cuda_push_array(*net->input_gpu, x, x_size);
    cuda_push_array(*net->truth_gpu, y, y_size);
  }
  state.input = *net->input_gpu;
  state.delta = 0;
  state.truth = *net->truth_gpu;
  state.train = 1;
#if defined(CUDNN_HALF) && defined(CUDNN)
  for (int i = 0; i < net->n; ++i)
  {
    layer* l = &net->layers[i];
    if (net->cudnn_half)
    {
      if (l->type == CONVOLUTIONAL && l->weights_gpu && l->weights_gpu16)
      {
        assert((l->nweights) > 0);
        cuda_convert_f32_to_f16(l->weights_gpu, l->nweights, l->weights_gpu16);
      }
    }
  }
#endif
  ForwardNetworkGpu(net, state);
  BackwardNetworkGpu(net, state);
}

float TrainNetworkDatumGpu(Network* net, float* x, float* y)
{
  *net->seen += net->batch;

  ForwardBackwardNetworkGpu(net, x, y);

  return GetNetworkCost(net);
}

typedef struct
{
  Network net;
  data d;
  float* err;
} train_args;

void* train_thread(void* ptr)
{
  train_args args = *(train_args*)ptr;
  free(ptr);
  cuda_set_device(args.net.gpu_index);
  *args.err = TrainNetwork(&args.net, args.d);
  return 0;
}

pthread_t train_network_in_thread(Network net, data d, float* err)
{
  pthread_t thread;
  train_args* ptr = (train_args*)calloc(1, sizeof(train_args));
  ptr->net = net;
  ptr->d = d;
  ptr->err = err;
  if (pthread_create(&thread, 0, train_thread, ptr))
    error("Thread creation failed");
  return thread;
}

void merge_weights(layer* l, layer* base)
{
  if (l->type == CONVOLUTIONAL)
  {
    axpy_cpu(l->n, 1, l->biases, 1, base->biases, 1);
    axpy_cpu(l->nweights, 1, l->weights, 1, base->weights, 1);
    if (l->scales)
    {
      axpy_cpu(l->n, 1, l->scales, 1, base->scales, 1);
    }
  }
  else if (l->type == CONNECTED)
  {
    axpy_cpu(l->outputs, 1, l->biases, 1, base->biases, 1);
    axpy_cpu(l->outputs * l->inputs, 1, l->weights, 1, base->weights, 1);
  }
}

void scale_weights(layer* l, float s)
{
  if (l->type == CONVOLUTIONAL)
  {
    scal_cpu(l->n, s, l->biases, 1);
    scal_cpu(l->nweights, s, l->weights, 1);
    if (l->scales)
      scal_cpu(l->n, s, l->scales, 1);
  }
  else if (l->type == CONNECTED)
  {
    scal_cpu(l->outputs, s, l->biases, 1);
    scal_cpu(l->outputs * l->inputs, s, l->weights, 1);
  }
}

void pull_weights(layer* l)
{
  if (l->type == CONVOLUTIONAL)
  {
    cuda_pull_array(l->biases_gpu, l->biases, l->n);
    cuda_pull_array(l->weights_gpu, l->weights, l->nweights);
    if (l->scales)
      cuda_pull_array(l->scales_gpu, l->scales, l->n);
  }
  else if (l->type == CONNECTED)
  {
    cuda_pull_array(l->biases_gpu, l->biases, l->outputs);
    cuda_pull_array(l->weights_gpu, l->weights, l->outputs * l->inputs);
  }
}

void distribute_weights(layer* l, layer* base)
{
  if (l->type == CONVOLUTIONAL)
  {
    cuda_push_array(l->biases_gpu, base->biases, l->n);
    cuda_push_array(l->weights_gpu, base->weights, l->nweights);
    if (base->scales)
      cuda_push_array(l->scales_gpu, base->scales, l->n);
  }
  else if (l->type == CONNECTED)
  {
    cuda_push_array(l->biases_gpu, base->biases, l->outputs);
    cuda_push_array(l->weights_gpu, base->weights, l->outputs * l->inputs);
  }
}

void sync_layer(Network* nets, int n, int j)
{
  // printf("Syncing layer %d\n", j);
  Network net = nets[0];
  layer* base = &net.layers[j];
  cuda_set_device(net.gpu_index);
  pull_weights(base);
  for (int i = 1; i < n; ++i)
  {
    cuda_set_device(nets[i].gpu_index);
    layer* l = &nets[i].layers[j];
    pull_weights(l);
    merge_weights(l, base);
  }
  scale_weights(base, 1. / n);
  for (int i = 0; i < n; ++i)
  {
    cuda_set_device(nets[i].gpu_index);
    layer* l = &nets[i].layers[j];
    distribute_weights(l, base);
  }
  // printf("Done syncing layer %d\n", j);
}

typedef struct
{
  Network* nets;
  int n;
  int j;
} sync_args;

void* sync_layer_thread(void* ptr)
{
  sync_args args = *(sync_args*)ptr;
  sync_layer(args.nets, args.n, args.j);
  free(ptr);
  return 0;
}

pthread_t sync_layer_in_thread(Network* nets, int n, int j)
{
  pthread_t thread;
  sync_args* ptr = (sync_args*)calloc(1, sizeof(sync_args));
  ptr->nets = nets;
  ptr->n = n;
  ptr->j = j;
  if (pthread_create(&thread, 0, sync_layer_thread, ptr))
    error("Thread creation failed");
  return thread;
}

void SyncNetworks(Network* nets, int num_gpus, int sync_interval)
{
  int layers = nets[0].n;
  int actual_batch = nets[0].batch * nets[0].subdiv;

  *nets[0].seen += sync_interval * (num_gpus - 1) * actual_batch;
  for (int j = 1; j < num_gpus; ++j)
  {
    *nets[j].seen = *nets[0].seen;
  }

  pthread_t* threads = (pthread_t*)calloc(layers, sizeof(pthread_t));
  for (int j = 0; j < layers; ++j)
  {
    threads[j] = sync_layer_in_thread(nets, num_gpus, j);
  }
  for (int j = 0; j < layers; ++j)
  {
    pthread_join(threads[j], 0);
  }
  free(threads);
}

float TrainNetworks(Network* nets, int num_gpus, data d, int sync_interval)
{
#ifdef _DEBUG
  int img_per_step = nets[0].batch * nets[0].subdiv * num_gpus;
  assert(img_per_step == d.X.rows);
#endif

  pthread_t* threads = (pthread_t*)calloc(num_gpus, sizeof(pthread_t));
  float* errors = (float*)calloc(num_gpus, sizeof(float));
  for (int i = 0; i < num_gpus; ++i)
  {
    data p = GetPartialData(d, i, num_gpus);
    threads[i] = train_network_in_thread(nets[i], p, errors + i);
  }

  float sum = 0;
  for (int i = 0; i < num_gpus; ++i)
  {
    pthread_join(threads[i], 0);
    sum += errors[i];
  }
  // cudaDeviceSynchronize();
  *nets[0].curr_iter += (num_gpus - 1);

  if (GetCurrIter(&nets[0]) % sync_interval == 0)
    SyncNetworks(nets, num_gpus, sync_interval);

  // cudaDeviceSynchronize();
  free(threads);
  free(errors);

  return (float)sum / num_gpus;
}

float* GetNetworkOutputLayerGpu(Network* net, int i)
{
  layer* l = &net->layers[i];
  cuda_pull_array(l->output_gpu, l->output, l->outputs * l->batch);

  return l->output;
}

float* GetNetworkOutputGpu(Network* net)
{
  int i;
  for (i = net->n - 1; i > 0; --i)
  {
    if (net->layers[i].type != COST)
      break;
  }

  return GetNetworkOutputLayerGpu(net, i);
}

float* NetworkPredictGpu(Network* net, float* input)
{
  if (net->gpu_index != cuda_get_device())
    cuda_set_device(net->gpu_index);
  int size = GetNetworkInputSize(net) * net->batch;

  NetworkState state;
  state.index = 0;
  state.net = net;
  state.input = net->input_state_gpu;
  memcpy(net->input_pinned_cpu, input, size * sizeof(float));
  cuda_push_array(state.input, net->input_pinned_cpu, size);
  state.truth = 0;
  state.train = 0;
  state.delta = 0;

  ForwardNetworkGpu(net, state);

  return GetNetworkOutputGpu(net);
}
