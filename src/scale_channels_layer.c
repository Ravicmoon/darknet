#include "scale_channels_layer.h"

#include <assert.h>
#include <stdio.h>

#include "activations.h"
#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"

layer make_scale_channels_layer(int batch, int index, int w, int h, int c,
    int w2, int h2, int c2, int scale_wh)
{
  fprintf(stderr, "scale Layer: %d\n", index);
  layer l = {(LAYER_TYPE)0};
  l.type = SCALE_CHANNELS;
  l.batch = batch;
  l.scale_wh = scale_wh;
  l.w = w;
  l.h = h;
  l.c = c;
  if (!l.scale_wh)
    assert(w == 1 && h == 1);
  else
    assert(c == 1);

  l.out_w = w2;
  l.out_h = h2;
  l.out_c = c2;
  if (!l.scale_wh)
    assert(l.out_c == l.c);
  else
    assert(l.out_w == l.w && l.out_h == l.h);

  l.outputs = l.out_w * l.out_h * l.out_c;
  l.inputs = l.outputs;
  l.index = index;

  l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
  l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

  l.forward = ForwardScaleChannelsLayer;
  l.backward = BackwardScaleChannelsLayer;
#ifdef GPU
  l.forward_gpu = ForwardScaleChannelsLayerGpu;
  l.backward_gpu = BackwardScaleChannelsLayerGpu;

  l.delta_gpu = cuda_make_array(l.delta, l.outputs * batch);
  l.output_gpu = cuda_make_array(l.output, l.outputs * batch);
#endif
  return l;
}

void ResizeScaleChannelsLayer(layer* l, Network* net)
{
  layer first = net->layers[l->index];
  l->out_w = first.out_w;
  l->out_h = first.out_h;
  l->outputs = l->out_w * l->out_h * l->out_c;
  l->inputs = l->outputs;
  l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
  l->output =
      (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
  cuda_free(l->output_gpu);
  cuda_free(l->delta_gpu);
  l->output_gpu = cuda_make_array(l->output, l->outputs * l->batch);
  l->delta_gpu = cuda_make_array(l->delta, l->outputs * l->batch);
#endif
}

void ForwardScaleChannelsLayer(layer* l, NetworkState state)
{
  int size = l->batch * l->out_c * l->out_w * l->out_h;
  int channel_size = l->out_w * l->out_h;
  int batch_size = l->out_c * l->out_w * l->out_h;
  float* from_output = state.net->layers[l->index].output;

  if (l->scale_wh)
  {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
      int input_index = i % channel_size + (i / batch_size) * channel_size;

      l->output[i] = state.input[input_index] * from_output[i];
    }
  }
  else
  {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
      l->output[i] = state.input[i / channel_size] * from_output[i];
    }
  }

  activate_array(l->output, l->outputs * l->batch, l->activation);
}

void BackwardScaleChannelsLayer(layer* l, NetworkState state)
{
  gradient_array(l->output, l->outputs * l->batch, l->activation, l->delta);

  int size = l->batch * l->out_c * l->out_w * l->out_h;
  int channel_size = l->out_w * l->out_h;
  int batch_size = l->out_c * l->out_w * l->out_h;
  float* from_output = state.net->layers[l->index].output;
  float* from_delta = state.net->layers[l->index].delta;

  if (l->scale_wh)
  {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
      int input_index = i % channel_size + (i / batch_size) * channel_size;

      state.delta[input_index] += l->delta[i] * from_output[i];
      from_delta[i] += state.input[input_index] * l->delta[i];
    }
  }
  else
  {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
      state.delta[i / channel_size] += l->delta[i] * from_output[i];
      from_delta[i] += state.input[i / channel_size] * l->delta[i];
    }
  }
}

#ifdef GPU
void ForwardScaleChannelsLayerGpu(layer* l, NetworkState state)
{
  int size = l->batch * l->out_c * l->out_w * l->out_h;
  int channel_size = l->out_w * l->out_h;
  int batch_size = l->out_c * l->out_w * l->out_h;

  scale_channels_gpu(state.net->layers[l->index].output_gpu, size, channel_size,
      batch_size, l->scale_wh, state.input, l->output_gpu);

  activate_array_ongpu(l->output_gpu, l->outputs * l->batch, l->activation);
}

void BackwardScaleChannelsLayerGpu(layer* l, NetworkState state)
{
  gradient_array_ongpu(
      l->output_gpu, l->outputs * l->batch, l->activation, l->delta_gpu);

  int size = l->batch * l->out_c * l->out_w * l->out_h;
  int channel_size = l->out_w * l->out_h;
  int batch_size = l->out_c * l->out_w * l->out_h;
  float* from_output = state.net->layers[l->index].output_gpu;
  float* from_delta = state.net->layers[l->index].delta_gpu;

  backward_scale_channels_gpu(l->delta_gpu, size, channel_size, batch_size,
      l->scale_wh, state.input, from_delta, from_output, state.delta);
}
#endif
