#pragma once

#include <sstream>
#include <vector>

class ArgValidator
{    
    // Description:
    // This class was created to compartmentalize the parsing of command line arguments.

    // Design Pattern:  Singleton

  private:
    const int m_argc;
    const char* const * const m_argv;
    std::stringstream m_argErrors;
    
    // The network topology defines the number of layers in the network and the number of neurons in each layer.
    // The input layer has 784 neurons (one for each pixel in an MNIST database 28x28 image) and the output layer 
    // has 10 neurons (from one-hot encoding each digit from 0 to 9). By default the network contains 2 hidden
    // layers, each with 16 neurons. This is a relatively small topology.
    std::vector<int> m_topology { 784, 16, 16, 10 };

    // The batch size indicates the number of training samples to backpropagate before updating the network's 
    // connection weights.
    int m_batchSize = 1;

    // An epoch refers to a single iteration of training a neural network with the entire training data set.
    int m_epochs = 1;

    // The learning rate serves as a multiplier for the weight gradient for each connection weight in the 
    // network. It is commonly represented as the greek letter alpha.
    double m_alpha = 0.0002;

    // The regularization parameter serves as a multiplier for the regularization component of the weight 
    // gradient for each connection weight. It is commonly represented as the greek letter lambda.
    double m_lambda = 0.0;

    // Neural networks with the same hyperparameters may demonstrate different performance after training.
    // This results from the radomization associated with initializing connection weights in the networks and
    // the order in which training data is fed into the networks. Users may select the number of networks to
    // train so they may assess how performance can vary.
    int m_numNetworks = 1;

  public:
    ~ArgValidator() = default;
    
    ArgValidator(const ArgValidator&)            = delete;
    ArgValidator(ArgValidator&&)                 = delete;
    ArgValidator& operator=(const ArgValidator&) = delete;
    ArgValidator& operator=(ArgValidator&&)      = delete;

    static void initInstance(const int argc, const char* const * const argv);
    static const ArgValidator& getInstance();

    constexpr const std::vector<int>&  getTopology() const { return m_topology; }
    constexpr int                      getBatchSize() const { return m_batchSize; }
    constexpr int                      getEpochs() const { return m_epochs; }
    constexpr double                   getAlpha() const { return m_alpha; }
    constexpr double                   getLambda() const { return m_lambda; }
    constexpr int                      getNumNetworks() const { return m_numNetworks; }
    constexpr const std::stringstream& getErrors() const { return m_argErrors; }

    static bool isNumeric(const char* const strSource, double& valResult);
    static bool isInteger(const double val);

  private:
    ArgValidator(const int argc, const char* const * const argv); // Make constructor inaccessible outside of class

    static const ArgValidator& getInstanceImpl(const int argc = 0, const char* const * const argv = nullptr);

    bool nextArgIsNumeric(const int argIndex, double& argVal);

    void getHyperparameterPosInt(int& hyperparameterIndex, int& hyperparameter);
    void getHyperparameterPosReal(int& hyperparameterIndex, double& hyperparameter);
    void getHyperparameterNonnegReal(int& hyperparameterIndex, double& hyperparameter);
};
