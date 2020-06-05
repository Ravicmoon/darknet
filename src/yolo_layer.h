#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer MakeYoloLayer(int batch, int w, int h, int n, int total, int* mask,
    int classes, int max_boxes);
void ForwardYoloLayer(const layer l, NetworkState state);
void BackwardYoloLayer(const layer l, NetworkState state);
void ResizeYoloLayer(layer* l, int w, int h);
int YoloNumDetections(layer l, float thresh);
int GetYoloDetections(layer l, int w, int h, int netw, int neth, float thresh,
    int* map, int relative, Detection* dets, int letter);
void CorrectYoloBoxes(Detection* dets, int n, int w, int h, int netw, int neth,
    int relative, int letter);

#ifdef GPU
void ForwardYoloLayerGpu(const layer l, NetworkState state);
void BackwardYoloLayerGpu(layer l, NetworkState state);
#endif

#ifdef __cplusplus
}
#endif
#endif
