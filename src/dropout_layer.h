#pragma once

#include "network.h"

void FillDropoutLayer(layer* l, int batch, int inputs, float probability,
    int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w,
    int h, int c);
void ResizeDropoutLayer(layer* l, int inputs);
void ForwardDropoutLayer(layer* l, NetworkState state);
void BackwardDropoutLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardDropoutLayerGpu(layer* l, NetworkState state);
void BackwardDropoutLayerGpu(layer* l, NetworkState state);
#endif