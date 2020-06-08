#pragma once

#include "network.h"

layer make_reorg_old_layer(
    int batch, int w, int h, int c, int stride, int reverse);
void ResizeReorgOldLayer(layer* l, int w, int h);
void ForwardReorgOldLayer(layer* l, NetworkState state);
void BackwardReorgOldLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardReorgOldLayerGpu(layer* l, NetworkState state);
void BackwardReorgOldLayerGpu(layer* l, NetworkState state);
#endif
