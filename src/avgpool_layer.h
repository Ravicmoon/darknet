#pragma once

#include "network.h"

layer make_avgpool_layer(int batch, int w, int h, int c);
void ResizeAvgpoolLayer(layer* l, int w, int h);
void ForwardAvgpoolLayer(layer* l, NetworkState state);
void BackwardAvgpoolLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardAvgpoolLayerGpu(layer* l, NetworkState state);
void BackwardAvgpoolLayerGpu(layer* l, NetworkState state);
#endif
