#pragma once

#include "network.h"

void FillAvgpoolLayer(layer* l, int batch, int w, int h, int c);
void ResizeAvgpoolLayer(layer* l, int w, int h);
void ForwardAvgpoolLayer(layer* l, NetworkState state);
void BackwardAvgpoolLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardAvgpoolLayerGpu(layer* l, NetworkState state);
void BackwardAvgpoolLayerGpu(layer* l, NetworkState state);
#endif
