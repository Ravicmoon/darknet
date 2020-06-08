#pragma once

#include "network.h"
#include "yolo_core.h"

layer make_gaussian_yolo_layer(int batch, int w, int h, int n, int total,
    int* mask, int classes, int max_boxes);
void ForwardGaussianYoloLayer(layer* l, NetworkState state);
void BackwardGaussianYoloLayer(layer* l, NetworkState state);
void ResizeGaussianYoloLayer(layer* l, int w, int h);
int GaussianYoloNumDetections(layer const* l, float thresh);
int GetGaussianYoloDetections(layer const* l, int w, int h, int netw, int neth,
    float thresh, int* map, int relative, Detection* dets, int letter);
void CorrectGaussianYoloBoxes(Detection* dets, int n, int w, int h, int netw,
    int neth, int relative, int letter);

#ifdef GPU
void ForwardGaussianYoloLayerGpu(layer* l, NetworkState state);
void BackwardGaussianYoloLayerGpu(layer* l, NetworkState state);
#endif
