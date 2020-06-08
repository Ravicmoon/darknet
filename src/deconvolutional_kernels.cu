#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "blas.h"
#include "col2im.h"
#include "convolutional_layer.h"
#include "dark_cuda.h"
#include "deconvolutional_layer.h"
#include "gemm.h"
#include "im2col.h"
#include "utils.h"

void forward_deconvolutional_layer_gpu(layer* l, NetworkState state)
{
  int out_w = DeconvolutionalOutWidth(l);
  int out_h = DeconvolutionalOutHeight(l);
  int size = out_w * out_h;

  int m = l->size * l->size * l->n;
  int n = l->h * l->w;
  int k = l->c;

  fill_ongpu(l->outputs * l->batch, 0, l->output_gpu, 1);

  for (int i = 0; i < l->batch; ++i)
  {
    float* a = l->weights_gpu;
    float* b = state.input + i * l->c * l->h * l->w;
    float* c = l->col_image_gpu;

    gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

    col2im_ongpu(c, l->n, out_h, out_w, l->size, l->stride, 0,
        l->output_gpu + i * l->n * size);
  }
  add_bias_gpu(l->output_gpu, l->biases_gpu, l->batch, l->n, size);
  activate_array(l->output_gpu, l->batch * l->n * size, l->activation);
}

void backward_deconvolutional_layer_gpu(layer* l, NetworkState state)
{
  float alpha = 1. / l->batch;
  int out_w = DeconvolutionalOutWidth(l);
  int out_h = DeconvolutionalOutHeight(l);
  int size = out_w * out_h;

  gradient_array(
      l->output_gpu, size * l->n * l->batch, l->activation, l->delta_gpu);
  backward_bias(l->bias_updates_gpu, l->delta, l->batch, l->n, size);

  if (state.delta)
    memset(state.delta, 0, l->batch * l->h * l->w * l->c * sizeof(float));

  for (int i = 0; i < l->batch; ++i)
  {
    int m = l->c;
    int n = l->size * l->size * l->n;
    int k = l->h * l->w;

    float* a = state.input + i * m * n;
    float* b = l->col_image_gpu;
    float* c = l->weight_updates_gpu;

    im2col_ongpu(l->delta_gpu + i * l->n * size, l->n, out_h, out_w, l->size,
        l->stride, 0, b);
    gemm_ongpu(0, 1, m, n, k, alpha, a, k, b, k, 1, c, n);

    if (state.delta)
    {
      int m = l->c;
      int n = l->h * l->w;
      int k = l->size * l->size * l->n;

      float* a = l->weights_gpu;
      float* b = l->col_image_gpu;
      float* c = state.delta + i * n * m;

      gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
  }
}

void update_deconvolutional_layer_gpu(
    layer* l, int skip, float learning_rate, float momentum, float decay)
{
  int size = l->size * l->size * l->c * l->n;

  axpy_ongpu(l->n, learning_rate, l->bias_updates_gpu, 1, l->biases_gpu, 1);
  scal_ongpu(l->n, momentum, l->bias_updates_gpu, 1);

  axpy_ongpu(size, -decay, l->weights_gpu, 1, l->weight_updates_gpu, 1);
  axpy_ongpu(size, learning_rate, l->weight_updates_gpu, 1, l->weights_gpu, 1);
  scal_ongpu(size, momentum, l->weight_updates_gpu, 1);
}

void pull_deconvolutional_layer(layer* l)
{
  cuda_pull_array(l->weights_gpu, l->weights, l->c * l->n * l->size * l->size);
  cuda_pull_array(l->biases_gpu, l->biases, l->n);
  cuda_pull_array(l->weight_updates_gpu, l->weight_updates,
      l->c * l->n * l->size * l->size);
  cuda_pull_array(l->bias_updates_gpu, l->bias_updates, l->n);
}

void push_deconvolutional_layer(layer* l)
{
  cuda_push_array(l->weights_gpu, l->weights, l->c * l->n * l->size * l->size);
  cuda_push_array(l->biases_gpu, l->biases, l->n);
  cuda_push_array(l->weight_updates_gpu, l->weight_updates,
      l->c * l->n * l->size * l->size);
  cuda_push_array(l->bias_updates_gpu, l->bias_updates, l->n);
}