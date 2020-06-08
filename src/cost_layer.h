#pragma once

#include "network.h"

COST_TYPE get_cost_type(char* s);
layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale);
void ResizeCostLayer(layer* l, int inputs);
void ForwardCostLayer(layer* l, NetworkState state);
void BackwardCostLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardCostLayerGpu(layer* l, NetworkState state);
void BackwardCostLayerGpu(layer* l, NetworkState state);
#endif
