#include "activation_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blas.h"
#include "dark_cuda.h"
#include "gemm.h"
#include "utils.h"

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
  layer l = {(LAYER_TYPE)0};
  l.type = ACTIVE;

  l.inputs = inputs;
  l.outputs = inputs;
  l.batch = batch;

  l.output = (float*)xcalloc(batch * inputs, sizeof(float));
  l.delta = (float*)xcalloc(batch * inputs, sizeof(float));

  l.forward = ForwardActivationLayer;
  l.backward = BackwardActivationLayer;
#ifdef GPU
  l.forward_gpu = ForwardActivationLayerGpu;
  l.backward_gpu = BackwardActivationLayerGpu;

  l.output_gpu = cuda_make_array(l.output, inputs * batch);
  l.delta_gpu = cuda_make_array(l.delta, inputs * batch);
#endif
  l.activation = activation;
  fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
  return l;
}

void ForwardActivationLayer(layer* l, NetworkState state)
{
  copy_cpu(l->outputs * l->batch, state.input, 1, l->output, 1);
  activate_array(l->output, l->outputs * l->batch, l->activation);
}

void BackwardActivationLayer(layer* l, NetworkState state)
{
  gradient_array(l->output, l->outputs * l->batch, l->activation, l->delta);
  copy_cpu(l->outputs * l->batch, l->delta, 1, state.delta, 1);
}

#ifdef GPU
void ForwardActivationLayerGpu(layer* l, NetworkState state)
{
  copy_ongpu(l->outputs * l->batch, state.input, 1, l->output_gpu, 1);
  activate_array_ongpu(l->output_gpu, l->outputs * l->batch, l->activation);
}

void BackwardActivationLayerGpu(layer* l, NetworkState state)
{
  gradient_array_ongpu(
      l->output_gpu, l->outputs * l->batch, l->activation, l->delta_gpu);
  copy_ongpu(l->outputs * l->batch, l->delta_gpu, 1, state.delta, 1);
}
#endif
