#ifndef SCALE_CHANNELS_LAYER_H
#define SCALE_CHANNELS_LAYER_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_scale_channels_layer(int batch, int index, int w, int h, int c,
    int w2, int h2, int c2, int scale_wh);
void forward_scale_channels_layer(const layer l, NetworkState state);
void backward_scale_channels_layer(const layer l, NetworkState state);
void resize_scale_channels_layer(layer* l, Network* net);

#ifdef GPU
void forward_scale_channels_layer_gpu(const layer l, NetworkState state);
void backward_scale_channels_layer_gpu(const layer l, NetworkState state);
#endif

#ifdef __cplusplus
}
#endif
#endif  // SCALE_CHANNELS_LAYER_H
