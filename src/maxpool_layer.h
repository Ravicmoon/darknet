#pragma once

#include "network.h"

void FillMaxpoolLayer(layer* l, int batch, int h, int w, int c, int size,
    int stride_x, int stride_y, int padding, int maxpool_depth,
    int out_channels, int antialiasing, int train);
void ResizeMaxpoolLayer(layer* l, int w, int h);
void ForwardMaxpoolLayer(layer* l, NetworkState state);
void BackwardMaxpoolLayer(layer* l, NetworkState state);

#ifdef GPU
void CudnnMaxpoolSetup(layer* l);
void ForwardMaxpoolLayerGpu(layer* l, NetworkState state);
void BackwardMaxpoolLayerGpu(layer* l, NetworkState state);
#endif  // GPU
