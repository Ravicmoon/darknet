#include "cost_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"

COST_TYPE GetCostType(char* s)
{
  if (strcmp(s, "sse") == 0)
    return SSE;
  if (strcmp(s, "masked") == 0)
    return MASKED;
  if (strcmp(s, "smooth") == 0)
    return SMOOTH;
  printf("Couldn't find cost type %s, going with SSE\n", s);
  return SSE;
}

void FillCostLayer(layer* l, int batch, int inputs, char* type_str, float scale)
{
  printf("cost                                           %4d\n", inputs);

  l->type = COST;
  l->scale = scale;
  l->batch = batch;
  l->inputs = inputs;
  l->outputs = inputs;
  l->cost_type = GetCostType(type_str);
  l->delta = (float*)xcalloc(inputs * batch, sizeof(float));
  l->output = (float*)xcalloc(inputs * batch, sizeof(float));
  l->cost = (float*)xcalloc(1, sizeof(float));

  l->forward = ForwardCostLayer;
  l->backward = BackwardCostLayer;
#ifdef GPU
  l->forward_gpu = ForwardCostLayerGpu;
  l->backward_gpu = BackwardCostLayerGpu;

  l->delta_gpu = cuda_make_array(l->delta, inputs * batch);
  l->output_gpu = cuda_make_array(l->output, inputs * batch);
#endif
}

void ResizeCostLayer(layer* l, int inputs)
{
  l->inputs = inputs;
  l->outputs = inputs;
  l->delta = (float*)xrealloc(l->delta, inputs * l->batch * sizeof(float));
  l->output = (float*)xrealloc(l->output, inputs * l->batch * sizeof(float));
#ifdef GPU
  cuda_free(l->delta_gpu);
  cuda_free(l->output_gpu);
  l->delta_gpu = cuda_make_array(l->delta, inputs * l->batch);
  l->output_gpu = cuda_make_array(l->output, inputs * l->batch);
#endif
}

void ForwardCostLayer(layer* l, NetworkState state)
{
  if (!state.truth)
    return;
  if (l->cost_type == MASKED)
  {
    int i;
    for (i = 0; i < l->batch * l->inputs; ++i)
    {
      if (state.truth[i] == SECRET_NUM)
        state.input[i] = SECRET_NUM;
    }
  }
  if (l->cost_type == SMOOTH)
  {
    smooth_l1_cpu(
        l->batch * l->inputs, state.input, state.truth, l->delta, l->output);
  }
  else
  {
    l2_cpu(l->batch * l->inputs, state.input, state.truth, l->delta, l->output);
  }
  l->cost[0] = sum_array(l->output, l->batch * l->inputs);
}

void BackwardCostLayer(layer* l, NetworkState state)
{
  axpy_cpu(l->batch * l->inputs, l->scale, l->delta, 1, state.delta, 1);
}

#ifdef GPU

void pull_cost_layer(layer l)
{
  cuda_pull_array(l.delta_gpu, l.delta, l.batch * l.inputs);
}

void push_cost_layer(layer l)
{
  cuda_push_array(l.delta_gpu, l.delta, l.batch * l.inputs);
}

int float_abs_compare(const void* a, const void* b)
{
  float fa = *(const float*)a;
  if (fa < 0)
    fa = -fa;
  float fb = *(const float*)b;
  if (fb < 0)
    fb = -fb;
  return (fa > fb) - (fa < fb);
}

void ForwardCostLayerGpu(layer* l, NetworkState state)
{
  if (!state.truth)
    return;
  if (l->cost_type == MASKED)
  {
    mask_ongpu(l->batch * l->inputs, state.input, SECRET_NUM, state.truth);
  }

  if (l->cost_type == SMOOTH)
  {
    smooth_l1_gpu(l->batch * l->inputs, state.input, state.truth, l->delta_gpu,
        l->output_gpu);
  }
  else
  {
    l2_gpu(l->batch * l->inputs, state.input, state.truth, l->delta_gpu,
        l->output_gpu);
  }

  if (l->ratio)
  {
    cuda_pull_array(l->delta_gpu, l->delta, l->batch * l->inputs);
    qsort(l->delta, l->batch * l->inputs, sizeof(float), float_abs_compare);
    int n = (1 - l->ratio) * l->batch * l->inputs;
    float thresh = l->delta[n];
    thresh = 0;
    printf("%f\n", thresh);
    supp_ongpu(l->batch * l->inputs, thresh, l->delta_gpu, 1);
  }

  cuda_pull_array(l->output_gpu, l->output, l->batch * l->inputs);
  l->cost[0] = sum_array(l->output, l->batch * l->inputs);
}

void BackwardCostLayerGpu(layer* l, NetworkState state)
{
  axpy_ongpu(l->batch * l->inputs, l->scale, l->delta_gpu, 1, state.delta, 1);
}
#endif
