#pragma once

#include "network.h"

layer MakeYoloLayer(int batch, int w, int h, int n, int total, int* mask,
    int classes, int max_boxes);
void ForwardYoloLayer(layer* l, NetworkState state);
void BackwardYoloLayer(layer* l, NetworkState state);
void ResizeYoloLayer(layer* l, int w, int h);
int YoloNumDetections(layer const* l, float thresh);
int GetYoloDetections(layer const* l, int w, int h, int netw, int neth,
    float thresh, int* map, int relative, Detection* dets, int letter);
void CorrectYoloBoxes(Detection* dets, int n, int w, int h, int netw, int neth,
    int relative, int letter);

#ifdef GPU
void ForwardYoloLayerGpu(layer* l, NetworkState state);
void BackwardYoloLayerGpu(layer* l, NetworkState state);
#endif
