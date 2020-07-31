#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "ArgValidator.h"
#include "MnistData.h"
#include "Network.h"
#include "Neuron.h"
#include "Timer.h"

constexpr static const char* const usageStr = R"(
usage:          Application.exe [-topology n1 [n2...]] [-batch_size b] [-n_epochs e] [-learning_rate lr] 
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
)";

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
)" <<

usageStr <<

R"(
The default network configuration is intended to be a simple demonstration. The parameters were chosen in the interest 
of a low network training time. Testing has shown an accuracy up to 85% is achievable with this configuration. Higher 
accuracies may be obtained with experimentation.

------------------------------------------------------------------------------------------------------------------------

Press Enter to close the program.
)";

    std::cin.get();
    return;
}

static void usage()
{
    std::cout << R"(
------------------------------------------------------------------------------------------------------------------------

help flags:     Application.exe [-help] [--help] [/help] [-h] [/h] [-?] [/?]

)" <<

usageStr <<

R"(
------------------------------------------------------------------------------------------------------------------------

)";

    return;
}

static std::mutex networksRemainingMutex;

static bool helpRequest(const int argc, const char* const * const argv);
static void trainAndTestNetworkMnist(Network& myNetwork, const MnistData& data, int& networksRemaining);

int main(const int argc, const char* const * const argv)
{       
    if (helpRequest(argc, argv)) {
        help();
        return 0;
    }

    std::cout << std::endl;
    
    if (argc == 1) { 
        // The default network configuration is intended to be a simple demonstration. The hyperparameters 
        // were chosen in the interest of a low network training time. Testing has shown an accuracy up to 
        // 85% is achievable with this configuration. Higher accuracies may be obtained with experimentation.
        std::cout << "Default training configuration:" << std::endl; 
    }
    else {  // Argument validation
        ArgValidator::initInstance(argc, argv);
        const ArgValidator& args = ArgValidator::getInstance();
        
        // If the arguments were successfully read, display the training configuration
        if (!std::strcmp(args.getErrors().str().c_str(), "")) {
            std::cout << "Custom training configuration:" << std::endl;
        }
        // Otherwise report the error and terminate the program
        else {
            usage();
            std::cerr << args.getErrors().str() << "Program terminated. Press Enter to close." << std::endl;
            std::cin.get();
            return -1;
        }
    }

    // Output the neural network configuration
    const ArgValidator& args = ArgValidator::getInstance();
    std::cout << "    Network Topology =         ";
    for (const int i : args.getTopology()) { 
        std::cout << i << " ";
    }
    std::cout << std::endl << 
        "    Batch Size =               " << args.getBatchSize() << std::endl << 
        "    Epochs =                   " << args.getEpochs() << std::endl <<
        "    Learning Rate =            " << args.getAlpha() << std::endl <<
        "    Regularization Parameter = " << args.getLambda() << std::endl <<
        "    Number of Networks =       " << args.getNumNetworks() << "\n" << std::endl;

    // Import the MNIST data
    const MnistData& data = MnistData::getInstance();
    if (!data.getDataValid()) {
        std::cerr << "[ERROR]: MINST data not available. Program terminated. Press Enter to close." << std::endl;
        std::cin.get();
        return -1;
    }

    // These hyperparameters will be shared among all neurons and are represented by static members of the Neuron class
    Neuron::setAlpha(args.getAlpha());
    Neuron::setLambda(args.getLambda());
    
    int n_networksRemaining = args.getNumNetworks();    // A count of the number of networks that still need 
                                                        // to be trained. Used for output to the console.

    std::vector<Network> myNetworks;    // The networks to be trained
    myNetworks.reserve(args.getNumNetworks());

    std::vector<std::future<void>> networkFutures; // Holds the return values from std::async
    networkFutures.reserve(args.getNumNetworks());

    // Train and test the networks asynchronously
    std::cout << "Training " << args.getNumNetworks() << " network(s)..." << std::endl;
    for (int i = 0; i < args.getNumNetworks(); i++) {
        myNetworks.emplace_back(args.getTopology(), args.getBatchSize(), args.getEpochs());
        networkFutures.emplace_back(std::async(std::launch::async, trainAndTestNetworkMnist, 
            std::ref(myNetworks.back()), std::ref(data), std::ref(n_networksRemaining)));
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

        if (maxAccuracy < metrics.accuracy) { 
            maxAccuracy = metrics.accuracy; 
        }

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

static bool helpRequest(const int argc, const char* const * const argv)
{
    constexpr const char* const helpSwitches[] {"-help", "--help", "/help", "-h", "/h", "-?", "/?"};
    for (int i = 1; i < argc; i++) {
        for (const char* const helpSwitch : helpSwitches) {
            if (!std::strcmp(argv[i], helpSwitch)) {
                return true;
            }
        }
    }

    return false;
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
    n_networksRemaining--;
    std::string notification = "Netowork trained! " + std::to_string(n_networksRemaining) + 
        (n_networksRemaining == 1 ? " network remains" : " networks remain") + ".\n";
    std::cout << notification;
}
