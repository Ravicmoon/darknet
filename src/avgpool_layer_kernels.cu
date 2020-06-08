#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "avgpool_layer.h"
#include "dark_cuda.h"

__global__ void forward_avgpool_layer_kernel(
    int n, int w, int h, int c, float* input, float* output)
{
  int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (id >= n)
    return;

  int k = id % c;
  id /= c;
  int b = id;

  int i;
  int out_index = (k + c * b);
  output[out_index] = 0;
  for (i = 0; i < w * h; ++i)
  {
    int in_index = i + h * w * (k + b * c);
    output[out_index] += input[in_index];
  }
  output[out_index] /= w * h;
}

__global__ void backward_avgpool_layer_kernel(
    int n, int w, int h, int c, float* in_delta, float* out_delta)
{
  int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (id >= n)
    return;

  int k = id % c;
  id /= c;
  int b = id;

  int i;
  int out_index = (k + c * b);
  for (i = 0; i < w * h; ++i)
  {
    int in_index = i + h * w * (k + b * c);
    in_delta[in_index] += out_delta[out_index] / (w * h);
  }
}

void ForwardAvgpoolLayerGpu(layer* l, NetworkState state)
{
  size_t n = l->c * l->batch;

  forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK, 0,
      get_cuda_stream()>>>(n, l->w, l->h, l->c, state.input, l->output_gpu);
  CHECK_CUDA(cudaPeekAtLastError());
}

void BackwardAvgpoolLayerGpu(layer* l, NetworkState state)
{
  size_t n = l->c * l->batch;

  backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK, 0,
      get_cuda_stream()>>>(n, l->w, l->h, l->c, state.delta, l->delta_gpu);
  CHECK_CUDA(cudaPeekAtLastError());
}
