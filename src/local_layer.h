#pragma once

#include "activations.h"
#include "network.h"

layer make_local_layer(int batch, int h, int w, int c, int n, int size,
    int stride, int pad, ACTIVATION activation);

void ForwardLocalLayer(layer* l, NetworkState state);
void BackwardLocalLayer(layer* l, NetworkState state);
void UpdateLocalLayer(
    layer* l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void ForwardLocalLayerGpu(layer* l, NetworkState state);
void BackwardLocalLayerGpu(layer* l, NetworkState state);
void UpdateLocalLayerGpu(layer* l, int batch, float learning_rate,
    float momentum, float decay, float loss_scale);

void PushLocalLayer(layer* l);
void PullLocalLayer(layer* l);
#endif