#pragma once

#include <algorithm>
#include <random>
#include <vector>

// Defined in this header file:
    class Neuron;
    typedef std::vector<Neuron> Layer;
    struct Connection;

class Neuron
{
    // Represents a neuron in a neural network. Handles the mathematics associated with feeding forward and backpropagating.
    
private:
    static std::random_device s_seed;
    static std::default_random_engine s_eng;
    static std::normal_distribution<double> s_generator;
    static double s_alpha;      // learning rate hyperparameter; typical range [0.0001 -> 0.01]
    static double s_lambda;     // regularization parameter hyperparameter

    std::vector<Connection> m_inputs;
    std::vector<Connection*> m_outputs;
    double m_inputVal           = 0.0;
    double m_outputVal          = 1.0;
    double m_activationGradient = 0.0;

public:
    Neuron(const Neuron& other);
    
    Neuron()              = default;
    ~Neuron()             = default;
    Neuron(Neuron&&)      = default;

    Neuron& operator=(const Neuron&) = delete;
    Neuron& operator=(Neuron&&)      = delete;

    constexpr static void setAlpha(const double alpha) { s_alpha = alpha; }
    constexpr static void setLambda(const double lambda) { s_lambda = lambda; }
    constexpr void setOutputVal(const double val) { m_outputVal = val; }
    constexpr double getOutputVal() const { return m_outputVal; }
    
    void initConnections(Layer& prevLayer, const int nextLayerSize, const int batchSize);
    void copyConnections(const Neuron& other, Layer& prevLayer);
    void feedForward();
    void calcOutputActivationGradient(const double targetVal);
    void calcHiddenActivationGradient();
    void calcInputWeightGradients();
    void updateInputWeights();

private:
    // ReLU was selected as the activation function
    constexpr double activationFunc()           const { return std::max(0.0, m_inputVal); }
    constexpr double activationFuncDerivative() const { return (m_inputVal > 0.0) ? 1.0 : 0.0; }
};

struct Connection
{
    // Represents a connection between two neurons in a neural network.
    
    const Neuron* const inputNeuron;           // A pointer to the connection's source neuron
    const Neuron* const outputNeuron;          // A pointer to the connection's destination neuron
    double weight;                             // The weight assigned to the connection
    std::vector<double> recentWeightGradients; // A vector containing a number of previous weight gradients 
                                               //      equal to the batch size. The mean of these values 
                                               //      is used when updating the connection weight. 
    int weightGradientIndex = 0;               // The index to which the next weight gradient will be assigned.
                                               //      It is reset to 0 once the end of a batch has been reached.
                                               //      Therefore, weight gradients from the previous batch will
                                               //      be overwritten as new weight gradients are calculated.
    
    Connection(const Neuron* const i, const Neuron* const o, const double w, const int batchSize);
    
    ~Connection()                 = default;
    Connection(Connection&&)      = default;

    Connection()                             = delete;
    Connection(const Connection&)            = delete;
    Connection& operator=(const Connection&) = delete;
    Connection& operator=(Connection&&)      = delete;
};
