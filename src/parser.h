#pragma once

#include "network.h"

void ParseNetworkCfg(Network* net, char const* filename, bool train = false);
void SaveWeights(Network* net, char const* filename);
void SaveWeightsUpTo(Network* net, char const* filename, int cutoff);
void LoadWeights(Network* net, char const* filename);
void LoadWeightsUpTo(Network* net, char const* filename, int cutoff);
