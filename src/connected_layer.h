#pragma once

#include "activations.h"
#include "network.h"

void FillConnectedLayer(layer* l, int batch, int steps, int inputs, int outputs,
    ACTIVATION activation, int batch_normalize);
void ForwardConnectedLayer(layer* l, NetworkState state);
void BackwardConnectedLayer(layer* l, NetworkState state);
void UpdateConnectedLayer(
    layer* l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void ForwardConnectedLayerGpu(layer* l, NetworkState state);
void BackwardConnectedLayerGpu(layer* l, NetworkState state);
void UpdateConnectedLayerGpu(layer* l, int batch, float learning_rate,
    float momentum, float decay, float loss_scale);
void PushConnectedLayer(layer* l);
void PullConnectedLayer(layer* l);
#endif
