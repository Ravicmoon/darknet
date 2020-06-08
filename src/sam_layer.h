#pragma once

#include "network.h"

layer make_sam_layer(
    int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void ResizeSamLayer(layer* l, int w, int h);
void ForwardSamLayer(layer* l, NetworkState state);
void BackwardSamLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardSamLayerGpu(layer* l, NetworkState state);
void BackwardSamLayerGpu(layer* l, NetworkState state);
#endif
