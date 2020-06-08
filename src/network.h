#pragma once

#include <stdint.h>

#include "data.h"
#include "yolo_core.h"

#ifdef GPU
float TrainNetworks(Network* nets, int n, data d, int interval);
void SyncNetworks(Network* nets, int n, int interval);
float TrainNetworkDatumGpu(Network* net, float* x, float* y);
float* NetworkPredictGpu(Network* net, float* input);
float* GetNetworkOutputGpu(Network* net);
void ForwardNetworkGpu(Network* net, NetworkState state);
void BackwardNetworkGpu(Network* net, NetworkState state);
void UpdateNetworkGpu(Network* net);
void ForwardBackwardNetworkGpu(Network* net, float* x, float* y);
#endif

float GetCurrentSeqSubdivisions(Network* net);
int GetSequenceValue(Network* net);
float GetCurrentRate(Network* net);
int GetCurrentBatch(Network* net);
int64_t GetCurrentIteration(Network* net);

void AllocateNetwork(Network* net, int n);
void ForwardNetwork(Network* net, NetworkState state);
void BackwardNetwork(Network* net, NetworkState state);
void UpdateNetwork(Network* net);

float TrainNetwork(Network* net, data d);
float TrainNetworkWaitKey(Network* net, data d, int wait_key);
float TrainNetworkBatch(Network* net, data d, int n);
float TrainNetworkSgd(Network* net, data d, int n);
float TrainNetworkDatum(Network* net, float* x, float* y);

float* GetNetworkOutput(Network* net);
int GetNetworkOutputSize(Network* net);
int ResizeNetwork(Network* net, int w, int h);
int GetNetworkInputSize(Network* net);
float GetNetworkCost(Network* net);

void copy_weights_net(Network net_train, Network* net_map);
