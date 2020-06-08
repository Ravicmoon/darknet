#pragma once

#include "network.h"

layer make_scale_channels_layer(int batch, int index, int w, int h, int c,
    int w2, int h2, int c2, int scale_wh);
void ResizeScaleChannelsLayer(layer* l, Network* net);
void ForwardScaleChannelsLayer(layer* l, NetworkState state);
void BackwardScaleChannelsLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardScaleChannelsLayerGpu(layer* l, NetworkState state);
void BackwardScaleChannelsLayerGpu(layer* l, NetworkState state);
#endif
