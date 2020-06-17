#pragma once

#include "activations.h"
#include "image.h"
#include "network.h"

#ifdef GPU
void ForwardConvolutionalLayerGpu(layer* l, NetworkState state);
void BackwardConvolutionalLayerGpu(layer* l, NetworkState state);
void UpdateConvolutionalLayerGpu(layer* l, int batch, float learning_rate,
    float momentum, float decay, float loss_scale);

void PushConvolutionalLayer(layer* l);
void PullConvolutionalLayer(layer* l);

void add_bias_gpu(float* output, float* biases, int batch, int n, int size);
void backward_bias_gpu(
    float* bias_updates, float* delta, int batch, int n, int size);
#ifdef CUDNN
void cudnn_convolutional_setup(
    layer* l, int cudnn_preference, size_t workspace_size_specify);
void create_convolutional_cudnn_tensors(layer* l);
void cuda_convert_f32_to_f16(float* input_f32, size_t size, float* output_f16);
#endif
#endif
void free_convolutional_batchnorm(layer* l);

size_t GetConvWorkspaceSize(layer* l);
void FillConvLayer(layer* l, int batch, int steps, int h, int w, int c, int n,
    int groups, int size, int stride_x, int stride_y, int dilation, int padding,
    ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam,
    int use_bin_output, int index, int antialiasing, layer* share_layer,
    int train);
void set_specified_workspace_limit(layer* l, size_t workspace_size_limit);
void resize_convolutional_layer(layer* layer, int w, int h);
void ForwardConvolutionalLayer(layer* l, NetworkState state);
void UpdateConvolutionalLayer(
    layer* l, int batch, float learning_rate, float momentum, float decay);
void binarize_weights(float* weights, int n, int size, float* binary);
void swap_binary(layer* l);
void binarize_weights2(
    float* weights, int n, int size, char* binary, float* scales);

void binary_align_weights(layer* l);

void BackwardConvolutionalLayer(layer* l, NetworkState state);

void add_bias(float* output, float* biases, int batch, int n, int size);
void backward_bias(
    float* bias_updates, float* delta, int batch, int n, int size);

int ConvOutHeight(layer* l);
int ConvOutWidth(layer* l);