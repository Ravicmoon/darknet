#pragma once

#include "network.h"

void FillUpsampleLayer(layer* l, int batch, int w, int h, int c, int stride);
void ResizeUpsampleLayer(layer* l, int w, int h);
void ForwardUpsampleLayer(layer* l, NetworkState state);
void BackwardUpsampleLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardUpsampleLayerGpu(layer* l, NetworkState state);
void BackwardUpsampleLayerGpu(layer* l, NetworkState state);
#endif
