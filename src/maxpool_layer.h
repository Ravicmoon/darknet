#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "dark_cuda.h"
#include "image.h"
#include "layer.h"
#include "network.h"

typedef layer maxpool_layer;

#ifdef __cplusplus
extern "C" {
#endif
Image get_maxpool_image(maxpool_layer l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size,
    int stride_x, int stride_y, int padding, int maxpool_depth,
    int out_channels, int antialiasing, int avgpool, int train);
void resize_maxpool_layer(maxpool_layer* l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, NetworkState state);
void backward_maxpool_layer(const maxpool_layer l, NetworkState state);

void forward_local_avgpool_layer(const maxpool_layer l, NetworkState state);
void backward_local_avgpool_layer(const maxpool_layer l, NetworkState state);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, NetworkState state);
void backward_maxpool_layer_gpu(maxpool_layer l, NetworkState state);
void cudnn_maxpool_setup(maxpool_layer* l);

void forward_local_avgpool_layer_gpu(maxpool_layer layer, NetworkState state);
void backward_local_avgpool_layer_gpu(maxpool_layer layer, NetworkState state);
#endif  // GPU

#ifdef __cplusplus
}
#endif

#endif
