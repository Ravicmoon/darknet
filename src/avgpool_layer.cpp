#include "avgpool_layer.h"

#include <stdio.h>

#include "dark_cuda.h"
#include "utils.h"

layer make_avgpool_layer(int batch, int w, int h, int c)
{
  fprintf(stderr, "avg                          %4d x%4d x%4d ->   %4d\n", w, h,
      c, c);
  layer l = {(LAYER_TYPE)0};
  l.type = AVGPOOL;
  l.batch = batch;
  l.h = h;
  l.w = w;
  l.c = c;
  l.out_w = 1;
  l.out_h = 1;
  l.out_c = c;
  l.outputs = l.out_c;
  l.inputs = h * w * c;
  int output_size = l.outputs * batch;
  l.output = (float*)xcalloc(output_size, sizeof(float));
  l.delta = (float*)xcalloc(output_size, sizeof(float));
  l.forward = ForwardAvgpoolLayer;
  l.backward = BackwardAvgpoolLayer;
#ifdef GPU
  l.forward_gpu = ForwardAvgpoolLayerGpu;
  l.backward_gpu = BackwardAvgpoolLayerGpu;
  l.output_gpu = cuda_make_array(l.output, output_size);
  l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif
  return l;
}

void ResizeAvgpoolLayer(layer* l, int w, int h)
{
  l->w = w;
  l->h = h;
  l->inputs = h * w * l->c;
}

void ForwardAvgpoolLayer(layer* l, NetworkState state)
{
  for (int b = 0; b < l->batch; ++b)
  {
    for (int k = 0; k < l->c; ++k)
    {
      int out_index = k + b * l->c;
      l->output[out_index] = 0;
      for (int i = 0; i < l->h * l->w; ++i)
      {
        int in_index = i + l->h * l->w * (k + b * l->c);
        l->output[out_index] += state.input[in_index];
      }
      l->output[out_index] /= l->h * l->w;
    }
  }
}

void BackwardAvgpoolLayer(layer* l, NetworkState state)
{
  for (int b = 0; b < l->batch; ++b)
  {
    for (int k = 0; k < l->c; ++k)
    {
      int out_index = k + b * l->c;
      for (int i = 0; i < l->h * l->w; ++i)
      {
        int in_index = i + l->h * l->w * (k + b * l->c);
        state.delta[in_index] += l->delta[out_index] / (l->h * l->w);
      }
    }
  }
}
