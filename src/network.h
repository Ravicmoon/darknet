#ifndef NETWORK_H
#define NETWORK_H
#include <stdint.h>

#include "data.h"
#include "image.h"
#include "layer.h"
#include "tree.h"
#include "yolo_core.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU
float TrainNetworks(Network* nets, int n, data d, int interval);
void sync_nets(Network* nets, int n, int interval);
float TrainNetworkDatumGpu(Network* net, float* x, float* y);
float* NetworkPredictGpu(Network* net, float* input);
float* get_network_delta_gpu_layer(Network net, int i);
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
void compare_networks(Network n1, Network n2, data d);
char* get_layer_string(LAYER_TYPE a);

void AllocateNetwork(Network* net, int n);
void ForwardNetwork(Network* net, NetworkState state);
void BackwardNetwork(Network* net, NetworkState state);
void UpdateNetwork(Network* net);

float TrainNetwork(Network* net, data d);
float TrainNetworkWaitKey(Network* net, data d, int wait_key);
float TrainNetworkBatch(Network* net, data d, int n);
float TrainNetworkSgd(Network* net, data d, int n);
float TrainNetworkDatum(Network* net, float* x, float* y);

matrix network_predict_data(Network net, data test);
float network_accuracy(Network net, data d);
float* network_accuracies(Network net, data d, int n);
float network_accuracy_multi(Network net, data d, int n);
void top_predictions(Network net, int n, int* index);
float* GetNetworkOutput(Network* net);
float* get_network_output_layer(Network net, int i);
float* get_network_delta_layer(Network net, int i);
float* get_network_delta(Network net);
int get_network_output_size_layer(Network net, int i);
int GetNetworkOutputSize(Network* net);
Image get_network_image(Network net);
Image get_network_image_layer(Network net, int i);
int get_predicted_class_network(Network net);
void print_network(Network net);
void visualize_network(Network net);
int resize_network(Network* net, int w, int h);
void set_batch_network(Network* net, int b);
int GetNetworkInputSize(Network* net);
float GetNetworkCost(Network* net);

int get_network_nuisance(Network net);
int get_network_background(Network net);
void copy_weights_net(Network net_train, Network* net_map);
void free_network_recurrent_state(Network net);
void randomize_network_recurrent_state(Network net);
void remember_network_recurrent_state(Network net);
void restore_network_recurrent_state(Network net);

#ifdef __cplusplus
}
#endif

#endif
