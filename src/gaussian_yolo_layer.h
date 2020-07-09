#pragma once

#include "network.h"

void FillGaussianYoloLayer(layer* l, int batch, int w, int h, int n, int total,
    int* mask, int classes, int max_boxes);
void ForwardGaussianYoloLayer(layer* l, NetworkState state);
void BackwardGaussianYoloLayer(layer* l, NetworkState state);
void ResizeGaussianYoloLayer(layer* l, int w, int h);
int GaussianYoloNumDetections(layer const* l, float thresh);
int GetGaussianYoloDetections(layer const* l, int w, int h, int netw, int neth,
    float thresh, int relative, Detection* dets);
void CorrectGaussianYoloBoxes(
    Detection* dets, int n, int w, int h, int netw, int neth, int relative);

#ifdef GPU
void ForwardGaussianYoloLayerGpu(layer* l, NetworkState state);
void BackwardGaussianYoloLayerGpu(layer* l, NetworkState state);
#endif
