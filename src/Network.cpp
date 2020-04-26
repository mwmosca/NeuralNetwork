#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "MnistData.h"
#include "Network.h"
#include "Timer.h"

std::random_device Network::s_seed;
std::default_random_engine Network::s_eng = std::default_random_engine(s_seed());

Network::Network(const std::vector<int>& topology, const int batchSize, const int epochs) 
    : m_batchSize(batchSize), m_epochs(epochs)
{
    // Set up a fully connected neural network based on the topology parameter
    // topology = {# neurons in layer 0, # neurons in layer 1, ..., # neurons in layer n-1}
    
    // Bias in this network will be modeled as an extra neuron in each layer. This neuron will have no input
    // neurons. It will have a constant activation of 1.0 (the default value of the Neuron member m_outputVal).
    
    if (topology.size() < 2) {
        std::cerr << "[ERROR]: The neural network must have at least an input layer and an output layer.";
        return;
    }
    
    m_layers.reserve(topology.size()); // Set up the layer vector

    // Add the input layer
    m_layers.emplace_back(Layer(topology[0] + 1));
    for (Neuron& j : m_layers.back()) {
        // The input layer has no previous layer so an empty Layer is 
        // passed as a parameter to initilize its neurons' connections
        j.initConnections(Layer(), topology[1], m_batchSize);
    }

    // Add the hidden layers
    for (int layerNum = 1; layerNum < topology.size() - 1; layerNum++) {
        m_layers.emplace_back();
        m_layers.back().reserve(topology[layerNum] + 1);
        for (int neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++) {
            m_layers.back().emplace_back();
            m_layers.back().back().initConnections(m_layers[layerNum - 1], topology[layerNum + 1], m_batchSize);
        }
        m_layers.back().emplace_back(); // Add a bias neuron for each hidden layer
        // The bias neuron will not receive input from the previous layer, so
        // an empty layer is passed as a parameter to initilize its connections.
        m_layers.back().back().initConnections(Layer(), topology[layerNum + 1], m_batchSize);
    }

    // Add the output layer
    m_layers.emplace_back();
    m_layers.back().reserve(topology.back() + 1);
    for (int neuronNum = 0; neuronNum < topology.back(); neuronNum++) {
        m_layers.back().emplace_back();
        m_layers.back().back().initConnections(m_layers[m_layers.size() - 2], 0, m_batchSize);
    }
    // Add a bias neuron for the output layer. This neuron will not be connected to any other neurons. 
    // It only serves as a placeholder to simplify the feed forward and back propagation algorithms.
    m_layers.back().emplace_back();
}

Network::Network(const Network& other)
    : m_layers(other.m_layers), m_batchSize(other.m_batchSize), m_batchCount(other.m_batchCount), 
      m_epochs(other.m_epochs), m_metrics(other.m_metrics)
{
    // Copy constructor

    // Make deep copies for connections in the input layer
    for (int neuronNum = 0; neuronNum < m_layers[0].size(); neuronNum++) {
        // The input layer has no previous layer so an empty Layer is 
        // passed as a parameter to initilize its neurons' connections
        m_layers[0][neuronNum].copyConnections(other.m_layers[0][neuronNum], Layer());
    }

    // Make deep copies for connections in the hidden layers
    for (int layerNum = 1; layerNum < m_layers.size(); layerNum++) {
        for (int neuronNum = 0; neuronNum < m_layers[layerNum].size(); neuronNum++) {
            m_layers[layerNum][neuronNum].copyConnections(other.m_layers[layerNum][neuronNum], m_layers[layerNum - 1]);
        }
    }
}

void Network::trainMnist(const MnistData& data)
{  
    // Train the neural network to be a classifier for images of digits from the MNIST database

    const Timer trainingTimer; // Start a timer to measure training performance

    // Alias the training data
    const int n_trainingSamples = data.getTrainingLabels().size();
    const std::vector<std::vector<double>>& trainingLabels = data.getTrainingLabels();
    const std::vector<std::vector<double>>& trainingImages = data.getTrainingImages();
    
    // Set up random indices for training data
    std::vector<int> randomIndices;
    randomIndices.reserve(n_trainingSamples);
    for (int i = 0; i < n_trainingSamples; i++) { randomIndices.push_back(i); }
    
    // Train the network
    for (int i = 0; i < m_epochs; i++) {
        std::shuffle(randomIndices.begin(), randomIndices.end(), s_eng); // Randomize the training data for each epoch
        for (int j = 0; j < n_trainingSamples; j++) {
            feedForward(trainingImages[randomIndices[j]]);
            backPropagate(trainingLabels[randomIndices[j]]);
        }
    }

    m_metrics.trainingTime = trainingTimer.getElapsedTime(Timer::UnitDivider::sec); // Record the training time
}

void Network::testMnist(const MnistData& data) 
{
    // Evaluate the neural network as a classifier for images of digits from the MNIST database
    
    // Alias the training data
    const int n_trainingSamples = data.getTrainingLabels().size();
    const std::vector<std::vector<double>>& trainingImages = data.getTrainingImages();
    const int n_testSamples = data.getTestLabels().size();
    const std::vector<double>& testLabels = data.getTestLabels();
    const std::vector<std::vector<double>>& testImages = data.getTestImages();

    // Set up a vector to extract results from the network
    const Layer& outputLayer = m_layers.back();
    std::vector<double> results;
    results.reserve(outputLayer.size() - 1); // Ignore the bias neuron
    
    // Count the number of misclassifications from the test set
    int misclassifications = 0;
    for (int i = 0; i < n_testSamples; i++) {
        feedForward(testImages[i]);
        
        // Extract the results
        results.clear();
        for (int neuronNum = 0; neuronNum < outputLayer.size() - 1; neuronNum++) {
            results.push_back(outputLayer[neuronNum].getOutputVal());
        }

        // results is a vector of scores for the digits 0-9. The highest score determines the image classification.
        int result = std::distance(results.begin(), std::max_element(results.begin(), results.end()));
        if (result != testLabels[i]) { misclassifications++; }
    }
    // Store the accuracy as a percentage
    m_metrics.accuracy = 100.0 * (n_testSamples - misclassifications) / n_testSamples;

    // Determine average processing time in milliseconds
    Timer processingTimer;
    for (const std::vector<double>& image : trainingImages) { feedForward(image); }
    for (const std::vector<double>& image : testImages)     { feedForward(image); }
    m_metrics.meanProcessingTime = 
        processingTimer.getElapsedTime(Timer::UnitDivider::msec) / (n_trainingSamples + n_testSamples);
}

void Network::feedForward(const std::vector<double>& inputVals)
{
    // Assign inputs to the input layer of the network and feed forward through each layer to the output layer
    
    // Set up the network inputs
    for (int inputNum = 0; inputNum < inputVals.size(); inputNum++) {
        m_layers[0][inputNum].setOutputVal(inputVals[inputNum]);
    }
    
    // Feed forward to update the activation for neurons in subsequent layers
    for (int layerNum = 1; layerNum < m_layers.size(); layerNum++) {
        // Bias neurons have constant output so skip them
        for (int neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++) {
            m_layers[layerNum][neuronNum].feedForward();
        }
    }
    
    return;
}

void Network::backPropagate(const std::vector<double>& targetVals)
{
    // Calculate the weight gradients for each connection and update the weights after each batch is processed
    
    // Calculate gradients for the output layer
    for (int outputNum = 0; outputNum < targetVals.size(); outputNum++) {
        m_layers.back()[outputNum].calcOutputActivationGradient(targetVals[outputNum]);
        m_layers.back()[outputNum].calcInputWeightGradients();
    }
    
    // Calculate gradients for the hidden layers
    for (int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
        // Bias neurons don't have input connections so skip them
        for (int neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++) {
            m_layers[layerNum][neuronNum].calcHiddenActivationGradient();
            m_layers[layerNum][neuronNum].calcInputWeightGradients();
        }
    }

    m_batchCount++;                         // Increment the number of samples processed in this batch
    if (m_batchCount == m_batchSize) {      // Check if the batch has been completed
        m_batchCount = 0;
        
        // Update the input weights for each neuron in the hidden layers and the output layer
        for (int layerNum = 1; layerNum < m_layers.size(); layerNum++) {
            // Bias neurons don't have input connections so skip them
            for (int neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++) {
                m_layers[layerNum][neuronNum].updateInputWeights();
            }
        }
    }

    return;
}
