#pragma once

#include <vector>

#include "Neuron.h"

class Network
{
    // Represents a neural network. Manages high level functionality such as organizing neurons into a structure
    // and issuing feed forward and backpropagation commands to individual neurons in the proper sequence.

    public:
        Network(const std::vector<unsigned>& topology, const unsigned t_batchSize);
        void feedForward(const std::vector<double>& inputVals);
        void backPropagate(const std::vector<double>& targetVals);
        void getResults(std::vector<double>& resultVals) const;
    private:
        std::vector<Layer> m_layers;
        const unsigned m_batchSize;
        unsigned m_batchCount;
};
