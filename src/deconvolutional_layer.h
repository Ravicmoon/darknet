#pragma once

#include "activations.h"
#include "network.h"

#ifdef GPU
void forward_deconvolutional_layer_gpu(layer* l, NetworkState state);
void backward_deconvolutional_layer_gpu(layer* l, NetworkState state);
void update_deconvolutional_layer_gpu(
    layer* l, int skip, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(layer* l);
void pull_deconvolutional_layer(layer* l);
#endif

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n,
    int size, int stride, ACTIVATION activation);
void ResizeDeconvolutionalLayer(layer* l, int h, int w);
void ForwardDeconvolutionalLayer(layer* l, NetworkState state);
void BackwardDeconvolutionalLayer(layer* l, NetworkState state);
void UpdateDeconvolutionalLayer(
    layer* l, int skip, float learning_rate, float momentum, float decay);

int DeconvolutionalOutWidth(layer* l);
int DeconvolutionalOutHeight(layer* l);
