#pragma once

#include "network.h"

layer make_dropout_layer(int batch, int inputs, float probability,
    int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w,
    int h, int c);

void ForwardDropoutLayer(layer* l, NetworkState state);
void BackwardDropoutLayer(layer* l, NetworkState state);
void ResizeDropoutLayer(layer* l, int inputs);

#ifdef GPU
void ForwardDropoutLayerGpu(layer* l, NetworkState state);
void BackwardDropoutLayerGpu(layer* l, NetworkState state);
#endif