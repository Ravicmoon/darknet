#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_yolo_layer(int batch, int w, int h, int n, int total, int* mask,
    int classes, int max_boxes);
void forward_yolo_layer(const layer l, NetworkState state);
void backward_yolo_layer(const layer l, NetworkState state);
void resize_yolo_layer(layer* l, int w, int h);
int YoloNumDetections(layer l, float thresh);
int yolo_num_detections_batch(layer l, float thresh, int batch);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh,
    int* map, int relative, Detection* dets, int letter);
int get_yolo_detections_batch(layer l, int w, int h, int netw, int neth,
    float thresh, int* map, int relative, Detection* dets, int letter,
    int batch);
void correct_yolo_boxes(Detection* dets, int n, int w, int h, int netw,
    int neth, int relative, int letter);

#ifdef GPU
void forward_yolo_layer_gpu(const layer l, NetworkState state);
void backward_yolo_layer_gpu(layer l, NetworkState state);
#endif

#ifdef __cplusplus
}
#endif
#endif
