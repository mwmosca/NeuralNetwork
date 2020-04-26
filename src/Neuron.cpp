#include <cmath>
#include <random>
#include <vector>

#include "Neuron.h"

// Set up a normally distributed random number generator with mean = 0.0 and stddev = 1.0
std::random_device Neuron::s_seed;
std::default_random_engine Neuron::s_eng = std::default_random_engine(s_seed());
std::normal_distribution<double> Neuron::s_generator = std::normal_distribution<double>(0.0, 1.0);

// Initialize hyperparameters
double Neuron::s_alpha = 0.0001;
double Neuron::s_lambda = 0.0;

Neuron::Neuron(const Neuron& other) 
    : m_inputVal(other.m_inputVal), m_outputVal(other.m_outputVal), m_activationGradient(other.m_activationGradient)
{
    // Copy constructor

    // The connection vectors are not copied. Deep copies for connections are set up in the copyConnections method.
}

void Neuron::initConnections(Layer& prevLayer, const int nextLayerSize, const int batchSize)
{   
    // Initialize the input connections for the calling neuron and "connect" 
    // this neuron to the relevant neurons in the previous layer.
    
    // Set up connection vectors for the calling neuron
    m_inputs.reserve(prevLayer.size());
    m_outputs.reserve(nextLayerSize);

    for (Neuron& k : prevLayer) {
        // Set up an input connection to the calling neuron coming from Neuron k in the previous layer.
        // Xavier initialization is used for the weights (ReLU variation):
        //      initial weight = randomNumber * sqrt(2 / neuronInputs)
        m_inputs.emplace_back(&k, this, s_generator(s_eng) * sqrt(2.0 / prevLayer.size()), batchSize);
        
        // Set up an output connection from neuron k in the previous layer to the calling neuron. This
        // is the same connection as the input connection set up above, so a pointer to that connection 
        // is stored.
        k.m_outputs.emplace_back(&m_inputs.back());
    }

    return;
}

void Neuron::copyConnections(const Neuron& other, Layer& prevLayer)
{
    // Create copies of the input connnections from Neuron other for the calling neuron. Deep copies are made
    // for pointer attributes. Then "connect" the calling neuron to the relevant neurons in the previous layer.
    
    // Set up connection vectors for the calling neuron
    m_inputs.reserve(other.m_inputs.size());
    m_outputs.reserve(other.m_outputs.size());
    
    for (int connectionNum = 0; connectionNum < other.m_inputs.size(); connectionNum++) {        
        // Set up aliases for ease of use
        Neuron& k = prevLayer[connectionNum];
        const Connection& otherConnection = other.m_inputs[connectionNum];
        
        // Set up an input connection to the calling neuron coming from Neuron k in the previous
        // layer. Non-pointer attributes will be copied from Connection otherConnection.
        m_inputs.emplace_back(&k, this, otherConnection.weight, (int) otherConnection.recentWeightGradients.size());
        m_inputs.back().weightGradientIndex = otherConnection.weightGradientIndex;

        // Set up an output connection from prevNeuron to the calling neuron. This is the same
        // connection as the input connection set up above, so a pointer to that connection is stored.
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
    //          = a_0*w_j0 + a_1*w_j1 + ... + a_n-1*w_jn-1 + a_n*w_jn + b_j
    //      a_k = the activation of neuron k in the previous layer
    //      w_jk = the weight assigned to the connection between neuron k in the previous layer and the calling neuron j
    //      b_j = the bias of the calling neuron j

    // Reminder: in this network architecture the bias b is represented as a neuron with a constant activation and output
    // connections with learnable weights.
    
    m_inputVal = 0.0;
    for (const Connection& c : m_inputs) { 
        m_inputVal += c.inputNeuron->m_outputVal * c.weight; 
    }
    m_outputVal = activationFunc();

    return;
}

void Neuron::calcOutputActivationGradient(const double targetVal)
{
    // Calculate the activation gradient of the cost function for the calling output neuron j:
    //      dC_i/da_j = 2(a_j - y_j)
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
    // Calculate the activation gradient of the cost function for the calling hidden neuron k:
    //      dC_i/da_k = (sum j = 1 -> n) (dC_i/da_j * da_j/dz_j * dz_j/da_k)
    //          where
    //      n = the number of neurons in the next layer connected to the calling hidden neuron k
    //      dC_i/da_j = the activation gradient of the cost function for neuron j in the next layer
    //      da_j/dz_j = the derivative of the activation function of neuron j in the next layer
    //      dz_j/da_k = the weight assigned to the connection between calling hidden neuron k and neuron j in the next layer

    // This activation gradient is later used to calculate the weight gradients for connections input into this neuron. 
    // Additionally, it is used to calculate the activation gradients for connected neurons in the previous layer.
    
    m_activationGradient = 0.0;
    for (const Connection* const c : m_outputs) {
        m_activationGradient += c->outputNeuron->m_activationGradient * // dC_i/da_j
            c->outputNeuron->activationFuncDerivative() *               // da_j/dz_j
            c->weight;                                                  // dz_j/da_k
    }

    return;
}

void Neuron::calcInputWeightGradients()
{
    // Calculate the weight gradient of the cost function for each input connection:
    //      dC_i/dw_jk = dC_i/da_j * da_j/dz_j * dz_j/dw_jk
    //          where 
    //      dC_i/da_j = the activation gradient of the cost function for calling neuron j
    //      da_j/dz_j = the derivative of the activation function of calling neuron j
    //      dz_j/dw_jk = the activation of neuron k in the previous layer

    // The weight gradients are used to update the connection weights after a batch of samples has been processed
    
    for (Connection& c : m_inputs) {
        c.recentWeightGradients[c.weightGradientIndex] = 
            m_activationGradient *       // dC_i/da_j
            activationFuncDerivative() * // da_j/dz_j
            c.inputNeuron->m_outputVal;  // dz_j/dw_jk
        
        c.weightGradientIndex++;    // Increment to the next weight gradient index
    }
        
    return;
}

void Neuron::updateInputWeights()
{
    // Update the weights of all input connections:
    //      w_jk = w_jk - alpha * dL/dw_jk
    //          where
    //      alpha = the learning rate hyperparameter
    //      dL/dw_jk = the weight gradient of the loss function
    //               = (1 / m) * ((sum i = 1 -> m) dC_i/dw_jk + lambda * w_jk)
    //      m = the batch size
    //      dC_i/dw_jk = the weight gradient of the cost function
    //      lambda = the regularization parameter hyperparameter
    
    for (Connection& c : m_inputs) {
        // Sum the weight gradients of the cost function in this batch
        double sumWeightGradients = 0.0;
        for (const double weightGradient : c.recentWeightGradients) { sumWeightGradients += weightGradient; }

        // Update the weight
        // w_jk = w_jk - alpha * dJ/dw_jk
        c.weight -= (s_alpha / c.recentWeightGradients.size()) * (sumWeightGradients + s_lambda * c.weight);

        c.weightGradientIndex = 0;      // Reset assignment index
    }

    return;
}

Connection::Connection(const Neuron* const i, const Neuron* const o, const double w, const int batchSize) 
        : inputNeuron(i), outputNeuron(o), weight(w), recentWeightGradients(batchSize, 0.0) {}
