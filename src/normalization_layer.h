#pragma once

#include "network.h"

layer make_normalization_layer(int batch, int w, int h, int c, int size,
    float alpha, float beta, float kappa);
void ResizeNormalizationLayer(layer* l, int w, int h);
void ForwardNormalizationLayer(layer* l, NetworkState state);
void BackwardNormalizationLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardNormalizationLayerGpu(layer* l, NetworkState state);
void BackwardNormalizationLayerGpu(layer* l, NetworkState state);
#endif
