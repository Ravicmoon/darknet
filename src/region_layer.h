#pragma once

#include "network.h"

layer make_region_layer(
    int batch, int w, int h, int n, int classes, int coords, int max_boxes);
void ResizeRegionLayer(layer* l, int w, int h);
void ForwardRegionLayer(layer* l, NetworkState state);
void BackwardRegionLayer(layer* l, NetworkState state);
void GetRegionBoxes(layer const* l, int w, int h, float thresh, float** probs,
    Box* boxes, int only_objectness, int* map);

#ifdef GPU
void ForwardRegionLayerGpu(layer* l, NetworkState state);
void BackwardRegionLayerGpu(layer* l, NetworkState state);
#endif
