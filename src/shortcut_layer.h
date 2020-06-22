#pragma once

#include "network.h"

void FillShortcutLayer(layer* l, int batch, int n, int* input_layers,
    int* input_sizes, int w, int h, int c, float** layers_output,
    float** layers_delta, float** layers_output_gpu, float** layers_delta_gpu,
    ACTIVATION activation, int train);
void ResizeShortcutLayer(layer* l, int w, int h, Network* net);
void ForwardShortcutLayer(layer* l, NetworkState state);
void BackwardShortcutLayer(layer* l, NetworkState state);

#ifdef GPU
void ForwardShortcutLayerGpu(layer* l, NetworkState state);
void BackwardShortcutLayerGpu(layer* l, NetworkState state);
void PushShortcutLayer(layer* l);
void PullShortcutLayer(layer* l);
#endif
