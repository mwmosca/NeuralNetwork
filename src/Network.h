#pragma once

#include <random>
#include <vector>

#include "MnistData.h"
#include "Neuron.h"

// Defined in this header file:
    struct NetworkMetrics;
    class Network;

struct NetworkMetrics
{
    // Performance measures for the network

    double accuracy           = 0.0;
    double trainingTime       = 0.0;
    double meanProcessingTime = 0.0;
    
    NetworkMetrics()                      = default;
    ~NetworkMetrics()                     = default;
    NetworkMetrics(const NetworkMetrics&) = default;
    NetworkMetrics(NetworkMetrics&&)      = default;

    NetworkMetrics& operator=(const NetworkMetrics&) = delete;
    NetworkMetrics& operator=(NetworkMetrics&&)      = delete;
};

class Network
{
    // Represents a neural network. Manages high level functionality such as organizing neurons into a structure
    // and issuing feed forward and backpropagation commands to individual neurons in the proper sequence.
    
private:
    static std::random_device s_seed;
    static std::default_random_engine s_eng;

    std::vector<Layer> m_layers;
    const int m_batchSize;
    int m_batchCount = 0;
    const int m_epochs;
    NetworkMetrics m_metrics;

public:
    Network(const std::vector<int>& topology, const int batchSize, const int epochs);
    Network(const Network& other);
    
    ~Network()         = default;
    Network(Network&&) = default;

    Network()                          = delete;
    Network& operator=(const Network&) = delete;
    Network& operator=(Network&&)      = delete;

    constexpr const NetworkMetrics& getMetrics() const { return m_metrics; }

    void trainMnist(const MnistData& data);
    void testMnist(const MnistData& data);

private:
    void feedForward(const std::vector<double>& inputVals);
    void backPropagate(const std::vector<double>& targetVals);
};
