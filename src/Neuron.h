#pragma once

#include <functional>
#include <random>
#include <vector>

class Neuron;
struct Connection;
typedef std::vector<Neuron> Layer;

class Neuron
{
    // Represents a neuron in a neural network. Handles the mathematics associated with feeding forward and backpropagating.
    
    public:
        Neuron();
        static void setAlpha(const double t_alpha) { s_alpha = t_alpha; }
        static void setLambda(const double t_lambda) { s_lambda = t_lambda; }
        double getOutoutVal() const { return m_outputVal; }
        void setOutputVal(const double val) { m_outputVal = val; }
        void initConnections(Layer& prevLayer, const unsigned nextLayerSize, const unsigned batchSize);
        void feedForward();
        void calcOutputActivationGradient(const double targetVal);
        void calcHiddenActivationGradient();
        void calcInputWeightGradients();
        void updateInputWeights();
    private:
        static std::random_device s_seed;
        static std::default_random_engine s_eng;
        static std::normal_distribution<double> s_generator;
        static double s_alpha;      // learning rate [0.0001 -> 0.01]
        static double s_lambda;     // regularization parameter
        std::vector<Connection> m_inputs;
        std::vector<Connection*> m_outputs;
        double m_inputVal;
        double m_outputVal;
        std::function<double(const double)> m_activationFunc;
        std::function<double(const double)> m_activationFuncDerivative;
        double m_activationGradient;
};

struct Connection
{
    // Represents a connection between two neurons in a neural network.
    
    Connection(const Neuron* const i, const Neuron* const o, const double w, const unsigned batchSize) 
        : inputNeuron(i), outputNeuron(o), weight(w), recentWeightGradients(batchSize, 0.0), weightGradientIndex(0) {}
    const Neuron* const inputNeuron;                // A pointer to the neuron that feeds the connetion
    const Neuron* const outputNeuron;               // A pointer to the neuron that accepts input from the connection
    double weight;                                  // The weight assigned to the connection
    std::vector<double> recentWeightGradients;      // A vector containing a number of previous weight gradients equal
                                                    //      to batchSize. The mean of these values is used when
                                                    //      updating the connection weight. 
    unsigned weightGradientIndex;                   // The index to which the next weight gradient will be assigned.
                                                    //      It is reset to 0 once the end of a batch has been reached.
                                                    //      Therefore, weight gradients from the previous batch will be
                                                    //      overwritten as new weight gradients are calculated.
};
