#pragma once

#include "network.h"

void FillReorgLayer(
    layer* l, int batch, int w, int h, int c, int stride, int reverse);
void resize_reorg_layer(layer* l, int w, int h);
void ForwardReorgLayer(layer* l, NetworkState state);
void BackwardReorgLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardReorgLayerGpu(layer* l, NetworkState state);
void BackwardReorgLayerGpu(layer* l, NetworkState state);
#endif
