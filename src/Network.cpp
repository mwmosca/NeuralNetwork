#pragma once

#include <vector>

#include "Network.h"

Network::Network(const std::vector<unsigned>& topology, const unsigned t_batchSize) 
    : m_batchSize(t_batchSize), m_batchCount(0)
{
    // Set up a fully connected neural network based on the topology parameter
    // topology = {# neurons in layer 0, # neurons in layer 1, ..., # neurons in layer n-1}
    
    // Bias in this network will be modeled as an extra neuron in each layer. This neuron will have no input neurons.
    // It will have a constant activation of 1.0 and output connections, each with a learnable weight.
    double biasNeuronOutput = 1.0;
    
    // Set up the layer vector
    m_layers.reserve(topology.size());

    // Add the input layer
    m_layers.emplace_back();
    m_layers.back().reserve(topology[0] + 1);
    for (unsigned neuronNum = 0; neuronNum < topology[0] + 1; neuronNum++) {
        m_layers.back().emplace_back();
        // The input layer has no previous layer so an empty layer is passed as a parameter to initilize its 
        // neuron's connections
        m_layers.back().back().initConnections(Layer(), topology[1], m_batchSize);
    }
    // Set the activation of the bias neuron for the input layer.
    m_layers.back().back().setOutputVal(biasNeuronOutput);
    
    // Add the hidden layers
    for (unsigned layerNum = 1; layerNum < topology.size() - 1; layerNum++) {
        m_layers.emplace_back();
        m_layers.back().reserve(topology[layerNum] + 1);
        for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++) {
            m_layers.back().emplace_back();
            m_layers.back().back().initConnections(m_layers[layerNum - 1], topology[layerNum + 1], m_batchSize);
        }
        // Add a bias neuron for each hidden layer
        m_layers.back().emplace_back();
        // The bias neuron will not receive input from the previous layer, so an empty layer is passed as a parameter 
        // to initilize its connections.
        m_layers.back().back().initConnections(Layer(), topology[layerNum + 1], m_batchSize);
        m_layers.back().back().setOutputVal(biasNeuronOutput);
    }

    // Add the output layer
    m_layers.emplace_back();
    m_layers.back().reserve(topology.back() + 1);
    for (unsigned neuronNum = 0; neuronNum < topology.back(); neuronNum++) {
        m_layers.back().emplace_back();
        m_layers.back().back().initConnections(m_layers[m_layers.size() - 2], 0, m_batchSize);
    }
    // Add a bias neuron for the output layer. This neuron will not be connected to any other neurons. It only serves
    // as a placeholder to simplify the feed forward and back propagation algorithms.
    m_layers.back().emplace_back();
}

void Network::feedForward(const std::vector<double>& inputVals)
{
    // Assign inputs to the input layer of the network and feed forward through each layer to the output layer
    
    // Set up the network inputs
    for (unsigned inputNum = 0; inputNum < inputVals.size(); inputNum++) {
        m_layers[0][inputNum].setOutputVal(inputVals[inputNum]);
    }
    
    // Feed forward to update the activation for neurons in subsequent layers
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
        // Bias neurons have constant output so skip them
        for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++) {
            m_layers[layerNum][neuronNum].feedForward();
        }
    
    return;
}

void Network::backPropagate(const std::vector<double>& targetVals)
{
    // Calculate the weight gradients for each connection and update the weights after each batch is processed.
    
    // Calculate gradients for the output layer
    for (unsigned neuronNum = 0; neuronNum < targetVals.size(); neuronNum++) {
        m_layers.back()[neuronNum].calcOutputActivationGradient(targetVals[neuronNum]);
        m_layers.back()[neuronNum].calcInputWeightGradients();
    }
    
    // Calculate gradients for the hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
        // Bias neurons don't have input connections so skip them
        for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++) {
            m_layers[layerNum][neuronNum].calcHiddenActivationGradient();
            m_layers[layerNum][neuronNum].calcInputWeightGradients();
        }
    }

    m_batchCount++;                         // Increment the number of samples processed in this batch
    if (m_batchCount == m_batchSize) {      // Check if the batch has been completed
        m_batchCount = 0;     // Reset the batch count
        
        // Update the input weights for each neuron in the hidden layers and the output layer
        for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
            // Bias neurons don't have input connections so skip them
            for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++) {
                m_layers[layerNum][neuronNum].updateInputWeights();
            }
        }
    }

    return;
}

void Network::getResults(std::vector<double>& resultVals) const
{
    // Stores the activations of the output layer in a results vector passed by reference.
    
    resultVals.clear();
    resultVals.reserve(m_layers.back().size() - 1);     // Ignore the bias neuron
    for (unsigned neuronNum = 0; neuronNum < m_layers.back().size() - 1; neuronNum++) {
        resultVals.emplace_back(m_layers.back()[neuronNum].getOutoutVal());
    }
    
    return;
}
