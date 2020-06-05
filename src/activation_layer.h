#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

void forward_activation_layer(layer l, NetworkState state);
void backward_activation_layer(layer l, NetworkState state);

#ifdef GPU
void forward_activation_layer_gpu(layer l, NetworkState state);
void backward_activation_layer_gpu(layer l, NetworkState state);
#endif

#ifdef __cplusplus
}
#endif

#endif
