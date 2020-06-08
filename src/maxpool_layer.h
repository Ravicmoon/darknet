#pragma once

#include "dark_cuda.h"
#include "network.h"

layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x,
    int stride_y, int padding, int maxpool_depth, int out_channels,
    int antialiasing, int avgpool, int train);
void ResizeMaxpoolLayer(layer* l, int w, int h);
void ForwardMaxpoolLayer(layer* l, NetworkState state);
void BackwardMaxpoolLayer(layer* l, NetworkState state);

void ForwardLocalAvgpoolLayer(layer* l, NetworkState state);
void BackwardLocalAvgpoolLayer(layer* l, NetworkState state);

#ifdef GPU
void CudnnMaxpoolSetup(layer* l);
void ForwardMaxpoolLayerGpu(layer* l, NetworkState state);
void BackwardMaxpoolLayerGpu(layer* l, NetworkState state);

void ForwardLocalAvgpoolLayerGpu(layer* layer, NetworkState state);
void BackwardLocalAvgpoolLayerGpu(layer* layer, NetworkState state);
#endif  // GPU
