#pragma once

#include "network.h"

layer make_detection_layer(int batch, int inputs, int n, int size, int classes,
    int coords, int rescore);
void ForwardDetectionLayer(layer* l, NetworkState state);
void BackwardDetectionLayer(layer* l, NetworkState state);
void GetDetectionDetections(
    layer const* l, int w, int h, float thresh, Detection* dets);

#ifdef GPU
void ForwardDetectionLayerGpu(layer* l, NetworkState state);
void BackwardDetectionLayerGpu(layer* l, NetworkState state);
#endif
