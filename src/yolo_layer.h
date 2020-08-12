#pragma once

#include "network.h"

void FillYoloLayer(layer* l, int batch, int w, int h, int n, int total,
    int* mask, int classes, int max_boxes);
void ForwardYoloLayer(layer* l, NetworkState state);
void BackwardYoloLayer(layer* l, NetworkState state);
void ResizeYoloLayer(layer* l, int w, int h);
int YoloNumDetections(layer const* l, float thresh);
int GetYoloDetections(
    layer const* l, int net_w, int net_h, float thresh, Detection* dets);

#ifdef GPU
void ForwardYoloLayerGpu(layer* l, NetworkState state);
void BackwardYoloLayerGpu(layer* l, NetworkState state);
#endif
