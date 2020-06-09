#pragma once

#include "network.h"

void FillCostLayer(
    layer* l, int batch, int inputs, char* type_str, float scale);
void ResizeCostLayer(layer* l, int inputs);
void ForwardCostLayer(layer* l, NetworkState state);
void BackwardCostLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardCostLayerGpu(layer* l, NetworkState state);
void BackwardCostLayerGpu(layer* l, NetworkState state);
#endif
