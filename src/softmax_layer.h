#pragma once

#include "network.h"

layer make_softmax_layer(int batch, int inputs, int groups);
void ForwardSoftmaxLayer(layer* l, NetworkState state);
void BackwardSoftmaxLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardSoftmaxLayerGpu(layer* l, NetworkState state);
void BackwardSoftmaxLayerGpu(layer* l, NetworkState state);
#endif
