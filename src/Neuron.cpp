#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "Neuron.h"

// Set up a normally distributed random number generator: mean = 0.0, stddev = 1.0
std::random_device Neuron::s_seed;
std::default_random_engine Neuron::s_eng = std::default_random_engine(s_seed());
std::normal_distribution<double> Neuron::s_generator = std::normal_distribution<double>(0.0, 1.0);

// Initialize hyperparameters
double Neuron::s_alpha = 0.0001;
double Neuron::s_lambda = 0.0;

Neuron::Neuron() : 
    m_activationFunc([] (const double x) {return std::max(0.0, x);}), 
    m_activationFuncDerivative([] (const double x) {return (x > 0.0) ? 1.0 : 0.0;})
{
    // Default to ReLU for activation function
    // To-do: overload constructor for other activation functions
}

void Neuron::initConnections(Layer& prevLayer, const unsigned nextLayerSize, const unsigned batchSize)
{   
    // Initialize the input connections for the calling neuron and "connect" 
    // this neuron to the relevant neurons in the previous layer.
    
    // Set up connection vectors for the calling neuron
    m_inputs.reserve(prevLayer.size());
    m_outputs.reserve(nextLayerSize);

    for (Neuron& k : prevLayer) {
        // Set up an input connection to the calling neuron coming from neuron k in the previous layer.
        // Xavier initialization is used for the weights (ReLU variation):
        //      connection weight = randomNumber * sqrt(2 / neuronInputs)
        m_inputs.emplace_back(&k, this, s_generator(s_eng) * sqrt(2.0/prevLayer.size()), batchSize);
        
        // Set up an output connection from neuron k in the previous layer to the calling neuron. This
        // is the same connection as the input connection set up above, so a pointer to that connection 
        // is stored.
        k.m_outputs.emplace_back(&m_inputs.back());
    }

    return;
}

void Neuron::feedForward() 
{
    // Calculate the activation of the calling neuron j:
    //      a_j = g(z_j)
    //          where
    //      g(z_j) = the activation function of the calling neuron j
    //      z_j = the input value of the calling neuron j
    //          = a_0*w_j0 + a_1*w_j1 + ... + a_n-1*w_jn-1 + a_n*w_jn + b
    //      a_k = the activation of neuron k in the previous layer
    //      w_jk = the weight assigned to the connection between neuron k in the previous layer and the calling neuron j
    //      b = the bias

    // Reminder: in this network architecture the bias b is represented as a neuron with a constant activation and output
    // connections with learnable weights.
    
    m_inputVal = 0.0;
    for (Connection& c : m_inputs) {
        m_inputVal += c.inputNeuron->m_outputVal * c.weight;
    }
    m_outputVal = m_activationFunc(m_inputVal);

    return;
}

void Neuron::calcOutputActivationGradient(const double targetVal)
{
    // Calculate the activation gradient of the loss function for the calling output neuron j:
    //      dL_i/da_j = 2(a_j - y_j)
    //          where
    //      a_j = the activation of the calling output neuron j
    //      y_j = the target activation of the calling output neuron j
    
    // This activation gradient is later used to calculate the weight gradients for connections input into this neuron. 
    // Additionally, it is used to calculate the activation gradients for connected neurons in the previous layer.

    m_activationGradient = 2 * (m_outputVal - targetVal);

    return;
}

void Neuron::calcHiddenActivationGradient() 
{
    // Calculate the activation gradient of the loss function for the calling hidden neuron k:
    //      dL_i/da_k = (sum j = 1 -> n) (dL_i/da_j * da_j/dz_j * dz_j/da_k)
    //          where
    //      n = the number of neurons in the next layer
    //      dL_i/da_j = the activation gradient of the loss function for neuron j in the next layer
    //      da_j/dz_j = the derivative of the activation function of neuron j in the next layer
    //      dz_j/da_k = the weight assigned to the connection between calling hidden neuron k and neuron j in the next layer

    // This activation gradient is later used to calculate the weight gradients for connections input into this neuron. 
    // Additionally, it is used to calculate the activation gradients for connected neurons in the previous layer.
    
    m_activationGradient = 0.0;
    for (Connection* c : m_outputs) {
        m_activationGradient += c->outputNeuron->m_activationGradient *                 // dL_i/da_j
            c->outputNeuron->m_activationFuncDerivative(c->outputNeuron->m_inputVal) *  // da_j/dz_j
            c->weight;                                                                  // dz_j/da_k
    }
    return;
}

void Neuron::calcInputWeightGradients()
{
    // Calculate the weight gradient of the loss function for each input connection:
    //      dL_i/dw_jk = dL_i/da_j * da_j/dz_j * dz_j/dw_jk
    //          where 
    //      dL_i/da_j = the activation gradient of the loss function for calling neuron j
    //      da_j/dz_j = the derivative of the activation function of calling neuron j
    //      dz_j/dw_jk = the activation of neuron k in the previous layer

    // The weight gradients are used to update the connection weights after a batch of samples has been processed
    
    for (Connection& c : m_inputs) {
        c.recentWeightGradients[c.weightGradientIndex] = 
            m_activationGradient *                      // dL_i/da_j
            m_activationFuncDerivative(m_inputVal) *    // da_j/dz_j
            c.inputNeuron->m_outputVal;                 // dz_j/dw_jk
        
        c.weightGradientIndex++;    // Increment to the next weight gradient index
    }
        
    return;
}

void Neuron::updateInputWeights()
{
    // Update the weights of all input connections:
    //      w_jk = w_jk - alpha * dJ/dw_jk
    //          where
    //      alpha = the learning rate hyperparameter
    //      dJ/dw_jk = the weight gradient of the cost function
    //               = (1 / m) * ((sum i = 1 -> m) dC_i/dw_jk + lambda * w_jk)
    //      m = the batch size
    //      dC_i/dw_jk = the weight gradient of the loss function
    //      lambda = the regularization parameter hyperparameter
    
    for (Connection& c : m_inputs) {
        // Sum the weight gradients of the loss function in this batch
        double sumWeightGradients = 0.0;
        for (double weightGradient : c.recentWeightGradients) sumWeightGradients += weightGradient;

        // Update the weight
        // w_jk = w_jk - alpha * dJ/dw_jk
        c.weight -= (s_alpha / c.recentWeightGradients.size()) * (sumWeightGradients + s_lambda * c.weight);

        c.weightGradientIndex = 0;      // Reset assignment index
    }

    return;
}
