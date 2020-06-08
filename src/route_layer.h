#pragma once

#include "network.h"

layer make_route_layer(int batch, int n, int* input_layers, int* input_size,
    int groups, int group_id);
void ResizeRouteLayer(layer* l, Network* net);
void ForwardRouteLayer(layer* l, NetworkState state);
void BackwardRouteLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardRouteLayerGpu(layer* l, NetworkState state);
void BackwardRouteLayerGpu(layer* l, NetworkState state);
#endif
