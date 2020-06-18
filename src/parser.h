#pragma once

#include "network.h"

void ParseNetworkCfg(Network* net, char const* filename, int batch = 0);
void SaveWeights(Network* net, char* filename);
void SaveWeightsUpTo(Network* net, char* filename, int cutoff);
void LoadWeights(Network* net, char const* filename);
void LoadWeightsUpTo(Network* net, char const* filename, int cutoff);
