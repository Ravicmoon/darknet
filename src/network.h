#pragma once

#include <stdint.h>

#include "data.h"
#include "yolo_core.h"

#ifdef GPU
float TrainNetworks(Network* nets, int num_gpus, data d, int sync_interval);
void SyncNetworks(Network* nets, int num_gpus);
float TrainNetworkDatumGpu(Network* net, float* x, float* y);
float* NetworkPredictGpu(Network* net, float* input);
float* GetNetworkOutputGpu(Network* net);
void ForwardNetworkGpu(Network* net, NetworkState state);
void BackwardNetworkGpu(Network* net, NetworkState state);
void UpdateNetworkGpu(Network* net);
void ForwardBackwardNetworkGpu(Network* net, float* x, float* y);
#endif

float GetCurrLr(Network* net);
int64_t GetCurrIter(Network* net);

void AllocateNetwork(Network* net, int n);
void ForwardNetwork(Network* net, NetworkState state);
void BackwardNetwork(Network* net, NetworkState state);
void UpdateNetwork(Network* net);

float TrainNetwork(Network* net, data d);
float TrainNetworkDatum(Network* net, float* x, float* y);

float* GetNetworkOutput(Network* net);
int GetNetworkInputSize(Network* net);
int GetNetworkOutputSize(Network* net);
void ResizeNetwork(Network* net, int w, int h);
float GetNetworkCost(Network* net);

void CopyNetWeights(Network* net_train, Network* net_map);
