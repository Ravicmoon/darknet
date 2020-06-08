#pragma once

#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c, int train);
void ResizeBatchnormLayer(layer* l, int w, int h);
void ForwardBatchnormLayer(layer* l, NetworkState state);
void BackwardBatchnormLayer(layer* l, NetworkState state);
void UpdateBatchnormLayer(
    layer* l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void ForwardBatchnormLayerGpu(layer* l, NetworkState state);
void BackwardBatchnormLayerGpu(layer* l, NetworkState state);
void UpdateBatchnormLayerGpu(layer* l, int batch, float learning_rate_init,
    float momentum, float decay, float loss_scale);
void PushBatchnormLayer(layer* l);
void PullBatchnormLayer(layer* l);
#endif
