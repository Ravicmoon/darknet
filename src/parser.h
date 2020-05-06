#ifndef PARSER_H
#define PARSER_H
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

void ParseNetworkCfg(Network* net, char const* filename);
void ParseNetworkCfgCustom(
    Network* net, char const* filename, int batch, int time_steps);
void save_network(Network net, char* filename);
void save_weights(Network net, char* filename);
void save_weights_upto(Network net, char* filename, int cutoff);
void save_weights_double(Network net, char* filename);
void LoadWeights(Network* net, char const* filename);
void LoadWeightsUpTo(Network* net, char const* filename, int cutoff);

#ifdef __cplusplus
}
#endif
#endif
