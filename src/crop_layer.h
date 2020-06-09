#pragma once

#include "network.h"

void FillCropLayer(layer* l, int batch, int h, int w, int c, int crop_height,
    int crop_width, int flip, float angle, float saturation, float exposure);
void ForwardCropLayer(layer* l, NetworkState state);
void ResizeCropLayer(layer* l, int w, int h);

#ifdef GPU
void ForwardCropLayerGpu(layer* l, NetworkState state);
#endif
