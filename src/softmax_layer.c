#include "softmax_layer.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"

#define SECRET_NUM -1234

void softmax_tree(float* input, int batch, int inputs, float temp,
    tree* hierarchy, float* output)
{
  int b;
  for (b = 0; b < batch; ++b)
  {
    int i;
    int count = 0;
    for (i = 0; i < hierarchy->groups; ++i)
    {
      int group_size = hierarchy->group_size[i];
      softmax(input + b * inputs + count, group_size, temp,
          output + b * inputs + count, 1);
      count += group_size;
    }
  }
}

layer make_softmax_layer(int batch, int inputs, int groups)
{
  assert(inputs % groups == 0);
  fprintf(
      stderr, "softmax                                        %4d\n", inputs);
  layer l = {(LAYER_TYPE)0};
  l.type = SOFTMAX;
  l.batch = batch;
  l.groups = groups;
  l.inputs = inputs;
  l.outputs = inputs;
  l.loss = (float*)xcalloc(inputs * batch, sizeof(float));
  l.output = (float*)xcalloc(inputs * batch, sizeof(float));
  l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
  l.cost = (float*)xcalloc(1, sizeof(float));

  l.forward = ForwardSoftmaxLayer;
  l.backward = BackwardSoftmaxLayer;
#ifdef GPU
  l.forward_gpu = ForwardSoftmaxLayerGpu;
  l.backward_gpu = BackwardSoftmaxLayerGpu;

  l.output_gpu = cuda_make_array(l.output, inputs * batch);
  l.loss_gpu = cuda_make_array(l.loss, inputs * batch);
  l.delta_gpu = cuda_make_array(l.delta, inputs * batch);
#endif
  return l;
}

void ForwardSoftmaxLayer(layer* l, NetworkState net)
{
  if (l->softmax_tree)
  {
    int i;
    int count = 0;
    for (i = 0; i < l->softmax_tree->groups; ++i)
    {
      int group_size = l->softmax_tree->group_size[i];
      softmax_cpu(net.input + count, group_size, l->batch, l->inputs, 1, 0, 1,
          l->temperature, l->output + count);
      count += group_size;
    }
  }
  else
  {
    softmax_cpu(net.input, l->inputs / l->groups, l->batch, l->inputs,
        l->groups, l->inputs / l->groups, 1, l->temperature, l->output);
  }

  if (net.truth && !l->noloss)
  {
    softmax_x_ent_cpu(
        l->batch * l->inputs, l->output, net.truth, l->delta, l->loss);
    l->cost[0] = sum_array(l->loss, l->batch * l->inputs);
  }
}

void BackwardSoftmaxLayer(layer* l, NetworkState net)
{
  axpy_cpu(l->inputs * l->batch, 1, l->delta, 1, net.delta, 1);
}

#ifdef GPU
void ForwardSoftmaxLayerGpu(layer* l, NetworkState net)
{
  if (l->softmax_tree)
  {
    softmax_tree_gpu(net.input, 1, l->batch, l->inputs, l->temperature,
        l->output_gpu, *l->softmax_tree);
  }
  else
  {
    if (l->spatial)
      softmax_gpu_new_api(net.input, l->c, l->batch * l->c, l->inputs / l->c,
          l->w * l->h, 1, l->w * l->h, 1, l->output_gpu);
    else
      softmax_gpu_new_api(net.input, l->inputs / l->groups, l->batch, l->inputs,
          l->groups, l->inputs / l->groups, 1, l->temperature, l->output_gpu);
  }
  if (net.truth && !l->noloss)
  {
    softmax_x_ent_gpu(l->batch * l->inputs, l->output_gpu, net.truth,
        l->delta_gpu, l->loss_gpu);
    if (l->softmax_tree)
    {
      mask_gpu_new_api(
          l->batch * l->inputs, l->delta_gpu, SECRET_NUM, net.truth, 0);
      mask_gpu_new_api(
          l->batch * l->inputs, l->loss_gpu, SECRET_NUM, net.truth, 0);
    }
    cuda_pull_array(l->loss_gpu, l->loss, l->batch * l->inputs);
    l->cost[0] = sum_array(l->loss, l->batch * l->inputs);
  }
}

void BackwardSoftmaxLayerGpu(layer* l, NetworkState net)
{
  axpy_ongpu(l->batch * l->inputs, 1, l->delta_gpu, 1, net.delta, 1);
}
#endif
