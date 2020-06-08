#pragma once

#include "activations.h"
#include "network.h"

layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

void ForwardActivationLayer(layer* l, NetworkState state);
void BackwardActivationLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardActivationLayerGpu(layer* l, NetworkState state);
void BackwardActivationLayerGpu(layer* l, NetworkState state);
#endif
