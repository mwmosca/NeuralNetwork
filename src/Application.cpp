#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "MnistData.h"
#include "Network.h"
#include "Neuron.h"
#include "Timer.h"

static void help()
{
    std::cout << R"(
------------------------------------------------------------------------------------------------------------------------

This program allows users to experiment with training a fully connected neural network to be a classifier for images of 
digits from the MNIST database (http://yann.lecun.com/exdb/mnist/). The network has 784 inputs (the number of pixels in 
each 28x28 image) and 10 outputs (from one-hot encoding each digit from 0 to 9). Users may set the following network 
hyperparameters:
    - number of hidden layers
    - number of neurons in each hidden layer
    - batch size
    - number of epochs
    - learning rate
    - regularization parameter

Neural networks with the same hyperparameters may demonstrate different performance after training. This results from 
the randomization associated with initializing connection weights in the networks and the order in which training data 
is fed into the networks. Users may select the number of networks to train so they may assess how performance can vary. 
After the networks are trained and tested the following model metrics are reported: 
    - maximum accuracy among all models
    - mean accuracy
    - mean training time
    - mean processing time

The code was written using the C++17 standard and compiled with Visual C++ 14.2.

usage:  neuralnetwork.exe   [-topology n1 [n2...]] [-batch_size b] [-n_epochs e] [-learning_rate lr] 
                            [-regularization_parameter rp] [-n_networks nets]

    -topology n1, n2, etc.:         positive integers indicating the number of 
                                    neurons in each hidden layer of the network
                                    (default topology is 16 16)

    -batch_size b:                  a positive integer indicating the number of training samples to 
                                    backpropagate before updating the network's connection weights
                                    (default batch_size is 1)

    -n_epochs e:                    a positive integer indicating the number of times the set of 
                                    training samples will be backpropagated through the network
                                    (default n_epochs is 1)

    -learning_rate lr:              a positive number that serves as a multiplier for the weight 
                                    gradient for each connection weight in the network
                                    (default learning_rate is 0.0002)

    -regularization_parameter rp:   a non-negative number that serves as a multiplier for the regularization 
                                    component of the weight gradient for each connection weight in the network
                                    (default regularization_parameter is 0.0)

    -n_networks nets:               a positive number indicating the number of networks 
                                    to train with the selected hyperparameters
                                    (default n_networks is 1)

The default network configuration is intended to be a simple demonstration. The parameters were chosen in the interest 
of a low network training time. Testing has shown an accuracy up to 85% is achievable with this configuration. Higher 
accuracies may be obtained with experimentation.

------------------------------------------------------------------------------------------------------------------------
)";
    
    return;
}

static std::mutex networksRemainingMutex;

static bool isNumeric(const char* const value);
static bool isInteger(const double value) { return std::nearbyint(value) == value; }
static void trainAndTestNetworkMnist(Network& myNetwork, const MnistData& data, int& networksRemaining);

int main(const int argc, const char** const argv)
{   
    constexpr const char* const helpSwitches[] {"-help", "--help", "/help", "-h", "/h", "-?", "/?"};
    constexpr int numHelpSwitches = 7;
    for (int i = 1; i < argc; i++) {
        for (const char* const helpSwitch : helpSwitches) {
            if (!std::strcmp(argv[i], helpSwitch)) {
                help();
                return 0;
            }
        }
    }

    std::cout << std::endl;
    
    // Set up the default network configuration. It is intended to be a simple demonstration. The following 
    // parameters were chosen in the interest of a low network training time. Testing has shown an accuracy 
    // up to 85% is achievable with this configuration. Higher accuracies may be obtained with experimentation.

    // The network topology defines the number of layers in the network and the number of neurons in each layer.
    // The input layer has 784 neurons (one for each pixel in an MNIST database 28x28 image) and the output layer 
    // has 10 neurons (from one-hot encoding each digit from 0 to 9). By default the network contains 2 hidden
    // layers, each with 16 neurons. This is a relatively small topology.
    std::vector<int> inputTopology {784, 16, 16, 10};

    // The batch size indicates the number of training samples to backpropagate before updating the network's 
    // connection weights.
    int inputBatchSize = 1;

    // An epoch refers to a single iteration of training a neural network with the entire training data set.
    int inputEpochs = 1;

    // The learning rate serves as a multiplier for the weight gradient for each connection weight in the 
    // network. It is commonly represented as the greek letter alpha.
    double inputAlpha = 0.0002;

    // The regularization parameter serves as a multiplier for the regularization component of the weight 
    // gradient for each connection weight. It is commonly represented as the greek letter lambda.
    double inputLambda = 0.0;

    // Neural networks with the same hyperparameters may demonstrate different performance after training.
    // This results from the radomization associated with initializing connection weights in the networks and
    // the order in which training data is fed into the networks. Users may select the number of networks to
    // train so they may assess how performance can vary.
    int inputNumNetworks = 1;
    
    if (argc == 1) { std::cout << "Default training configuration:" << std::endl; }
    
    else { // Argument validation
        std::stringstream argErrors; // A stringstream will be used to record argument errors
        
        double numericArgValidator;  // Used to check the qualities of a numeric argument:
                                     //      - negative, zero, or positvie
                                     //      - integer or double

        for (int i = 1; i < argc; i++) { // Iterate through arguments
            
            if (!std::strcmp(argv[i], "-topology")) {                      // Read in the network topology
                // Delete the default topology but keep the input layer
                inputTopology.erase(inputTopology.begin() + 1, inputTopology.end());

                if (!(i + 1 < argc) || !isNumeric(argv[i + 1])) { // Check if a topology argument is present and numeric
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no values were provided for -topology.\n";
                }

                for (; i + 1 < argc; i++) { // Read in topology arguments
                    // If a non-numeric argument is read break out of the topology reading loop
                    if (!isNumeric(argv[i + 1])) { break; }

                    numericArgValidator = std::atof(argv[i + 1]);
                    if (numericArgValidator > 0 && isInteger(numericArgValidator)) {
                        inputTopology.push_back(numericArgValidator);
                    }
                    else {
                        argErrors << "Error at argument index " << i + 1 << ", value " << numericArgValidator << 
                            ": -topology values must be positive integers.\n";
                    }
                }

                inputTopology.push_back(10); // Add the output layer to the topology
            }

            else if (!std::strcmp(argv[i], "-batch_size")) {               // Read in the batch size
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) { // Check if the batch size argument is present and numeric
                    i++;
                    numericArgValidator = std::atof(argv[i]);
                    if (numericArgValidator > 0 && isInteger(numericArgValidator)) {
                        inputBatchSize = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -batch_size must be a positive integer.\n";
                    }
                }
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -batch_size.\n";
                }
            }

            else if (!std::strcmp(argv[i], "-n_epochs")) {                 // Read in the number of epochs
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) { // Check if the epochs argument is present and numeric
                    i++;
                    numericArgValidator = std::atof(argv[i]);
                    if (numericArgValidator > 0 && isInteger(numericArgValidator)) {
                        inputEpochs = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -n_epochs must be a positive integer.\n";
                    }
                }
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -n_epochs.\n";
                }
            }

            else if (!std::strcmp(argv[i], "-learning_rate")) {            // Read in the learning rate
                // Check if the learning rate argument is present and numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) { 
                    i++;
                    numericArgValidator = std::atof(argv[i]);
                    if (numericArgValidator > 0) {
                        inputAlpha = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -learning_rate must be a positive number.\n";
                    }
                }
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -learning_rate.\n";
                }
            }

            else if (!std::strcmp(argv[i], "-regularization_parameter")) { // Read in the regularization parameter
                // Check if the regularization parameter argument is present and numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);
                    if (!(numericArgValidator < 0)) {
                        inputLambda = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -regularization_parameter must be a non-negative number.\n";
                    }
                }
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -regularization_parameter.\n";
                }
            }

            else if (!std::strcmp(argv[i], "-n_networks")) {               // Read in the number of networks
                // Check if the n_networks argument is present and whether it is numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);
                    if (numericArgValidator > 0 && isInteger(numericArgValidator)) {
                        inputNumNetworks = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -n_networks must be a positive integer.\n";
                    }
                }
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -n_networks.\n";
                }
            }

            else {
                argErrors << "[ERROR]: Argument index " << i << ", value " << argv[i] << ": argument not recognized.\n";
            }
        }

        // If the arguments were successfully read, display the training configuration
        if (!std::strcmp(argErrors.str().c_str(), "")) {
            std::cout << "Custom training configuration:" << std::endl;
        }
        // Otherwise report the error and terminate the program
        else {
            help();
            std::cerr << argErrors.str() << "Program terminated. Press Enter to close." << std::endl;
            std::cin.get();
            return -1;
        }
    }

    // Output the neural network configuration
    std::cout << "    Network Topology =         ";
    for (int i = 0; i < inputTopology.size(); i++) { std::cout << inputTopology[i] << " "; }
    std::cout << std::endl << 
        "    Batch Size =               " << inputBatchSize << std::endl << 
        "    Epochs =                   " << inputEpochs << std::endl <<
        "    Learning Rate =            " << inputAlpha << std::endl <<
        "    Regularization Parameter = " << inputLambda << std::endl <<
        "    Number of Networks =       " << inputNumNetworks << "\n" << std::endl;

    // Import the MNIST data
    const MnistData& data = MnistData::get();
    if (!data.getDataValid()) {
        std::cerr << "[ERROR]: MINST data not available. Program terminated. Press Enter to close." << std::endl;
        std::cin.get();
        return -1;
    }

    // Set up the neural network hyperparameters
    const std::vector<int> networkTopology(std::move(inputTopology));
    const int batchSize = inputBatchSize;
    const int epochs = inputEpochs;
    Neuron::setAlpha(inputAlpha);   // These hyperparameters are represented by static members of the Neuron class
    Neuron::setLambda(inputLambda);

    const int n_networks = inputNumNetworks; // The number of networks to be trained
    int n_networksRemaining = n_networks;    // A count of the number of networks that still need 
                                             //     to be trained. Used for output to the console.

    std::vector<Network> myNetworks;               // The networks to be trained
    myNetworks.reserve(n_networks);

    std::vector<std::future<void>> networkFutures; // Holds the return values from std::async
    networkFutures.reserve(n_networks);

    std::cout << "Training " << n_networks << " network(s)..." << std::endl;
    for (int networkNum = 0; networkNum < n_networks; networkNum++) {
        myNetworks.emplace_back(networkTopology, batchSize, epochs);
        networkFutures.emplace_back(std::async(std::launch::async, trainAndTestNetworkMnist, 
            std::ref(myNetworks[networkNum]), std::ref(data), std::ref(n_networksRemaining)));
    }
    for (const std::future<void>& f : networkFutures) f.wait(); // Wait for all network training to complete
    std::cout << std::endl;

    // Assess performance data
    double maxAccuracy = 0.0;
    double meanAccuracy = 0.0;
    double meanTrainingTime = 0.0;
    double meanProcessingTime = 0.0;

    for (const Network& net : myNetworks) {
        const NetworkMetrics& metrics = net.getMetrics();
        if (maxAccuracy < metrics.accuracy) { maxAccuracy = metrics.accuracy; }
        meanAccuracy += metrics.accuracy;
        meanTrainingTime += metrics.trainingTime;
        meanProcessingTime += metrics.meanProcessingTime;
    }

    meanAccuracy /= myNetworks.size();
    meanTrainingTime /= myNetworks.size();
    meanProcessingTime /= myNetworks.size();

    // Output performance data
    std::cout << "Maximum accuracy:\t" << maxAccuracy << " %" << std::endl <<
        "Mean Accuracy:\t\t" << meanAccuracy << " %" << std::endl <<
        "Mean Training Time:\t" << meanTrainingTime << " seconds" << std::endl <<
        "Mean Processing Time:\t" << meanProcessingTime << " milliseconds\n" << std::endl;

    std::cout << "Press Enter to close the program." << std::endl;
    std::cin.get();

    return 0;
}

static bool isNumeric(const char* const value)
{
    // Determines if value is a numeric type. Used for reading command line arguments.

    std::stringstream valueReader;
    double numericValidator;

    // If value can be successfully streamed into numericValidator it is numeric
    valueReader << value;
    valueReader >> numericValidator;

    return (bool) valueReader;
}

static void trainAndTestNetworkMnist(
    Network& net,            // The network to be trainined and tested
    const MnistData& data,   // MNIST data for training and testing the network
    int& n_networksRemaining // A count of the number of networks that still need to be trained. When asynchronously 
                             //      training multiple networks this count is shared among threads.
) {
    // Trains the neural network passed by reference to be a classifier for images of digits from the MNIST database.
    // May be called asynchronously to train multiple networks simultaneously.

    net.trainMnist(data);
    net.testMnist(data);

    // Log progress
    std::lock_guard<std::mutex> lock(networksRemainingMutex);
    std::string notification = "Netowork trained! " + std::to_string(--n_networksRemaining) + " networks remain.\n";
    std::cout << notification;
}
