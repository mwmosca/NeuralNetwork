#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// OpenCV was used for a rudimentary image display function during testing. It has been commented out of the deployed
// project for accessibility. Developers with OpenCV installed may uncomment this code if they wish to experiment with it.
// #include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>

#include "MnistData.cpp"
#include "Network.cpp"
#include "Neuron.cpp"
#include "Timer.cpp"

static void help()
{
    std::cout <<
        "\n--------------------------------------------------------------------------------------------\n" << std::endl <<

        "This program allows users to experiment with training a fully connected neural network to " << std::endl << 
        "be a classifier for images of digits from the MNIST database (http://yann.lecun.com/exdb/mnist/). " << std::endl <<
        "The network has 784 inputs (the number of pixels in each image (28x28)) and 10 outputs " << std::endl <<
        "(from one-hot encoding each digit from 0 to 9). Users may set the following network " << std::endl <<
        "hyperparameters:" << std::endl <<
        "    - number of hidden layers" << std::endl <<
        "    - number of neurons in each hidden layer" << std::endl <<
        "    - batch size" << std::endl <<
        "    - number of epochs" << std::endl <<
        "    - learning rate" << std::endl <<
        "    - regularization parameter\n" << std::endl <<

        "Neural networks with the same hyperparameters may demonstrate different performance after " << std::endl <<
        "training. This results from the randomization associated with initializing connection " << std::endl <<
        "weights in the networks and the order in which training data is fed into the networks. " << std::endl <<
        "Users may select the number of networks to train so they may assess how performance can " << std::endl <<
        "vary. After the networks are trained and tested the following model metrics are reported: " << std::endl <<
        "    - maximum accuracy among all models" << std::endl <<
        "    - mean training time" << std::endl <<
        "    - mean accuracy" << std::endl <<
        "    - mean processing time\n" << std::endl <<

        "The code was written using the C++ Standard Library. The only exception is a simple " << std::endl <<
        "function that displays MNIST images using OpenCV libraries. This has been commented out " << std::endl <<
        "of the source code for accessibility. The code was compiled with Visual C++ 14.2.\n" << std::endl <<

        "usage: neuralnetwork.exe [-topology n1 [n2...]] [-batch_size b] [-n_epochs e] " << std::endl <<
        "       [-learning_rate lr] [-regularization_parameter rp] [-n_networks nets]\n" << std::endl <<

        "    -topology n1, n2, etc.:       positive integers indicating the number of neurons in " << std::endl << 
        "                                  each hidden layer of the network" << std::endl <<
        "                                  (default topology is 16 16)\n" << std::endl <<

        "    -batch_size b:                a positive integer indicating the number of training " << std::endl <<
        "                                  samples to backpropagate before updating the network's " << std::endl <<
        "                                  connection weights" << std::endl <<
        "                                  (default batch_size is 1)\n" << std::endl <<

        "    -n_epochs e:                  a positive integer indicating the number of times the " << std::endl <<
        "                                  set of training samples will be backpropagated through " << std::endl <<
        "                                  the network" << std::endl <<
        "                                  (default n_epochs is 1)\n" << std::endl <<

        "    -learning_rate lr:            a positive number that serves as a multiplier for the " << std::endl <<
        "                                  weight gradient for each connection weight in the network " << std::endl <<
        "                                  (default learning_rate is 0.0002)\n" << std::endl <<

        "    -regularization_parameter rp: a non-negative number that serves as a multiplier for " << std::endl <<
        "                                  the regularization component of the weight gradient " << std::endl <<
        "                                  for each connection weight in the network" << std::endl <<
        "                                  (default regularization_parameter is 0.0)\n" << std::endl <<

        "    -n_networks nets:             a positive number indicating the number of networks to " << std::endl <<
        "                                  train with the selected hyperparameters" << std::endl <<
        "                                  (default n_networks is 1)\n" << std::endl <<

        "The default network configuration is intended to be a simple demonstration. The parameters " << std::endl <<
        "were chosen in the interest of a low network training time. Testing has shown an accuracy " << std::endl <<
        "up to 85% is achievable with this configuration. Higher accuracies may be obtained with " << std::endl <<
        "experimentation.\n" << std::endl <<

        "--------------------------------------------------------------------------------------------\n" << std::endl;
    
    return;
}

static std::mutex networksRemainingMutex;

static bool isNumeric(const char* const value);
static void trainAndTestNetworkMnist(Network* const myNetwork, const unsigned epochs, unsigned* const networksRemaining, 
    const MnistData* const data, std::vector<double>* const networkMetrics);
static void displayImage(const std::vector<double>& imageData);

int main(const int argc, const char** argv)
{   
    // Check for help request
    const char* helpSwitches[] = {"-help", "--help", "/help", "-h", "/h", "-?", "/?"};
    unsigned numHelpSwitches = 7;
    for (unsigned i = 1; i < argc; i++) {
        for (unsigned j = 0; j < numHelpSwitches; j++) {
            if (!std::strcmp(argv[i], helpSwitches[j])) {
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
    // The input layer has 784 neurons (one for each pixel in an MNIST database image (28x28)) and the output layer 
    // has 10 neurons (from one-hot encoding each digit from 0 to 9). By default the network contains 2 hidden
    // layers, each with 16 neurons. This is a relatively small topology.
    std::vector<unsigned> inputTopology = {784, 16, 16, 10};

    // The batch size indicates the number of training samples to backpropagate before updating the network's 
    // connection weights.
    unsigned inputBatchSize = 1;

    // An epoch refers to a single iteration of training a neural network with the entire training data set.
    unsigned inputEpochs = 1;

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
    unsigned inputNumNetworks = 1;

    std::vector<int> intInputs;
    
    if (argc == 1) {
        std::cout << "Default training configuration:" << std::endl;
    }
    
    // Argument validation
    else {
        std::stringstream argErrors;    // A stringstream will be used to record argument errors
        
        // Used to check the qualities of a numeric argument:
        //      - negative, zero, or positvie
        //      - integer or double
        double numericArgValidator;

        // Iterate through the arguments
        for (unsigned i = 1; i < argc; i++) {
            
            // Read in the network topology
            if (!std::strcmp(argv[i], "-topology")) {
                // Delete the default topology but keep the input layer
                inputTopology.erase(inputTopology.begin() + 1, inputTopology.end());

                // Check if a topology argument is present and whether it is numeric.
                // If no topology arguments were provided, record an error.
                if (!(i + 1 < argc) || !isNumeric(argv[i + 1])) {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no values were provided for -topology.\n";
                }

                // Read in arguments until a non-numeric is reached
                for (; i + 1 < argc; i++) {
                    // If a non-numeric argument is read break out of the topology reading loop
                    if (!isNumeric(argv[i + 1])) break;

                    numericArgValidator = std::atof(argv[i + 1]);

                    // If the value is a positive integer add it to the topology
                    if (numericArgValidator > 0 && std::nearbyint(numericArgValidator) == numericArgValidator) {
                        // A cast is used to avoid compiler warnings
                        inputTopology.emplace_back((unsigned) numericArgValidator);
                    }
                    else {
                        argErrors << "Error at argument index " << i + 1 << ", value " << numericArgValidator << 
                            ": -topology values must be positive integers.\n";
                    }
                }

                // Add the output layer to the topology
                inputTopology.emplace_back(10);
            }

            // Read in the batch size
            else if (!std::strcmp(argv[i], "-batch_size")) {
                // Check if the batch size argument is present and whether it is numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);

                    // If the value is a positive integer set the batch size
                    if (numericArgValidator > 0 && std::nearbyint(numericArgValidator) == numericArgValidator) {
                        inputBatchSize = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -batch_size must be a positive integer.\n";
                    }
                }

                // If no batch size argument was provided record an error
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -batch_size.\n";
                }
            }

            // Read in the number of epochs
            else if (!std::strcmp(argv[i], "-n_epochs")) {
                // Check if the epochs argument is present and whether it is numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);

                    // If the value is a positive integer set the number of epochs
                    if (numericArgValidator > 0 && std::nearbyint(numericArgValidator) == numericArgValidator) {
                        inputEpochs = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -n_epochs must be a positive integer.\n";
                    }
                }

                // If no epochs argument was provided record an error
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -n_epochs.\n";
                }
            }

            // Read in the learning rate
            else if (!std::strcmp(argv[i], "-learning_rate")) {
                // Check if the learning rate argument is present and whether it is numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);

                    // If the value is a positive number set the learning rate
                    if (numericArgValidator > 0) {
                        inputAlpha = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -learning_rate must be a positive number.\n";
                    }
                }

                // If no learning rate argument was provided record an error
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -learning_rate.\n";
                }
            }

            // Read in the regularization parameter
            else if (!std::strcmp(argv[i], "-regularization_parameter")) {
                // Check if the regularization parameter argument is present and whether it is numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);

                    // If the value is a non-negative number set the regularization parameter
                    if (!(numericArgValidator < 0)) {
                        inputLambda = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -regularization_parameter must be a non-negative number.\n";
                    }
                }

                // If no regularization parameter argument was provided record an error
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -regularization_parameter.\n";
                }
            }

            // Read in the number of networks
            else if (!std::strcmp(argv[i], "-n_networks")) {
                // Check if the networks argument is present and whether it is numeric
                if ((i + 1 < argc) && isNumeric(argv[i + 1])) {
                    i++;
                    numericArgValidator = std::atof(argv[i]);

                    // If the value is a positive integer set the number of networks
                    if (numericArgValidator > 0 && std::nearbyint(numericArgValidator) == numericArgValidator) {
                        inputNumNetworks = numericArgValidator;
                    }
                    else {
                        argErrors << "Error at argument index " << i << ", value " << numericArgValidator << 
                            ": -n_networks must be a positive integer.\n";
                    }
                }

                // If no networks argument was provided record an error
                else {
                    argErrors << "Error at argument index " << i << ", value " << argv[i] << 
                            ": no value was provided for -n_networks.\n";
                }
            }

            // An unrecognized argument was encountered. Record error message.
            else {
                argErrors << "Error at argument index " << i << ", value " << argv[i] << ": argument not recognized.\n";
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

    // Output neural network configuration
    std::cout << "    Network Topology =         ";
    for (unsigned i = 0; i < inputTopology.size(); i++) std::cout << inputTopology[i] << " ";
    std::cout << std::endl << 
        "    Batch Size =               " << inputBatchSize << std::endl << 
        "    Epochs =                   " << inputEpochs << std::endl <<
        "    Learning Rate =            " << inputAlpha << std::endl <<
        "    Regularization Parameter = " << inputLambda << std::endl <<
        "    Number of Networks =       " << inputNumNetworks << "\n" << std::endl;

    // Import the MNIST data
    MnistData data;
    if (!data.getDataValid()) {
        std::cerr << "MINST data not available. Program terminated. Press Enter to close." << std::endl;
        std::cin.get();
        return -1;
    }

    // Start a timer to measure performance
    Timer programTimer;

    // Set up the neural network hyperparameters
    const std::vector<unsigned> networkTopology(inputTopology);
    const unsigned batchSize = inputBatchSize;
    const unsigned epochs = inputEpochs;
    Neuron::setAlpha(inputAlpha);       // These hyperparameters are represented as static members of the Neuron class
    Neuron::setLambda(inputLambda);

    const unsigned numNetworks = inputNumNetworks;  // The number of networks to be trained
    unsigned networksRemaining = numNetworks;       // A count of the number of networks that still need to be trained.
                                                    // Used for output to the console.

    // The networks to be trained
    std::vector<Network> myNetworks;
    myNetworks.reserve(numNetworks);

    // Each element of networkMetricsCollection is a vector containing an individual network's metrics:
    //      {training time, accuracy, mean processing time}
    std::vector<std::vector<double>> networkMetricsCollection;
    networkMetricsCollection.reserve(numNetworks);

    // Holds the std::future<void> return values from std::async
    std::vector<std::future<void>> networkFutures;
    networkFutures.reserve(numNetworks);

    // Train the networks
    std::cout << "Training network(s)..." << std::endl;
    for (unsigned i = 0; i < numNetworks; i++) {
        myNetworks.emplace_back(networkTopology, batchSize);    // Create a new network
        networkMetricsCollection.emplace_back();                // Create a new metrics vector
        
        // Set up the network to be trained asynchronously
        networkFutures.emplace_back(std::async(std::launch::async, trainAndTestNetworkMnist, 
            &myNetworks[i], epochs, &networksRemaining, &data, &networkMetricsCollection[i]));
    }
    for (std::future<void>& i : networkFutures) i.wait();   // Wait for all network training to complete
    std::cout << std::endl;
    double programExecutionTime = programTimer.getElapsedSeconds();

    // Process performance data
    
    // Stores the maximum accuracy among trained networks
    double maxAcc = 0.0;
    
    // Used to extract the means from the network metrics:
    //      {mean training time, mean accuracy, mean processing time}
    std::vector<double> dataMeans(3, 0.0);

    for (unsigned i = 0; i < networkMetricsCollection.size(); i++) {
        if (networkMetricsCollection[i][1] > maxAcc) maxAcc = networkMetricsCollection[i][1];
        for (unsigned j = 0; j < networkMetricsCollection[i].size(); j++) dataMeans[j] += networkMetricsCollection[i][j];
    }
    for (unsigned i = 0; i < dataMeans.size(); i++) dataMeans[i] /= networkMetricsCollection.size();

    // Output performance data
    std::cout << "Maximum accuracy:\t" << maxAcc << " %" << std::endl;
    std::cout << "Mean Training Time:\t" << dataMeans[0] << " seconds" << std::endl;
    std::cout << "Mean Accuracy:\t\t" << dataMeans[1] << " %" << std::endl;
    std::cout << "Mean Processing Time:\t" << dataMeans[2] << " milliseconds\n" << std::endl;

    // Output performance data to a csv file
    // This was used during testing for ease of data collection.
    /*
    std::ofstream outputStream("output.csv");
    if (outputStream.is_open()) {
        outputStream << "Program Execution Time," << programExecutionTime << "\n";
        std::string rowHeaders[] = {"Training Time", "Accuracy", "Mean Processing Time"};
        for (unsigned i = 0; i < networkMetricsCollection[0].size(); i++) {
            outputStream << rowHeaders[i] << ",";
            for (unsigned j = 0; j < networkMetricsCollection.size(); j++) {}
                outputStream << networkMetricsCollection[j][i] << ",";
            }
            outputStream << "\n";
        }
        outputStream.close();
    }
    else std::cout << "The output file could not be opened." << std::endl;
    */

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
    Network* const myNetwork,                       // The network to be trainined and assessed
    const unsigned epochs,                          // The number of times the training set will be backpropagated through
                                                    //      the network
    unsigned* const networksRemaining,              // A count of the number of networks that still need to be trained.
                                                    //      When asynchronously training multiple networks this count is
                                                    //      shared among threads.
    const MnistData* const data,                    // MNIST data for training and testing the network
    std::vector<double>* const networkMetrics       // Container to store the metrics of the trained network
) {
    // Trains the neural network passed by reference to be a classifier for images of digits from the MNIST database.
    // Stores network metrics from the trained network in a vector passed by reference. May be called asynchronously 
    // to train multiple networks simultaneously.
    
    // Metrics of interest include:
    //      1) the time required to train the network.
    //      2) the accuracy with which the network can classify the test set.
    //      3) the mean processing time of the network.
    networkMetrics->reserve(3);

    // Set up randomization for training data
    std::vector<unsigned> indices;
    indices.reserve(data->m_trainingLabels.size());
    for (unsigned i = 0; i < data->m_trainingLabels.size(); i++) indices.emplace_back(i);
    std::random_device seed;
    std::default_random_engine eng(seed());

    // Start a timer to measure performance
    Timer trainingTimer;
    
    // Train the network
    for (unsigned i = 0; i < epochs; i++) {
        // Randomize the training data for each epoch
        std::shuffle(indices.begin(), indices.end(), eng);
        for (unsigned j = 0; j < data->m_trainingLabels.size(); j++) {
            myNetwork->feedForward(data->m_trainingImages[indices[j]]);
            myNetwork->backPropagate(data->m_trainingLabels[indices[j]]);
        }
    }
    // Store the training time
    networkMetrics->emplace_back(trainingTimer.getElapsedSeconds());

    // Assess the test data
    std::vector<double> networkResults;
    
    // Count the number of misclassifications from the test set
    unsigned misclassifications = 0;
    for (unsigned i = 0; i < data->m_testImages.size(); i++) {
        myNetwork->feedForward(data->m_testImages[i]);
        myNetwork->getResults(networkResults);
        
        // networkResults is a vector of scores for the digits 0-9. The highest score determines the image classification.
        if (std::distance(networkResults.begin(), std::max_element(networkResults.begin(), networkResults.end())) != 
            data->m_testLabels[i]) 
        {
            misclassifications++;
        }
    }
    // Store the accuracy as a percentage
    networkMetrics->emplace_back(100.0 * (data->m_testLabels.size() - misclassifications) / data->m_testLabels.size());

    // Determine average processing time
    Timer processingTimer;
    for (const std::vector<double>& image : data->m_trainingImages) myNetwork->feedForward(image);
    for (const std::vector<double>& image : data->m_testImages) myNetwork->feedForward(image);
    networkMetrics->emplace_back(
        processingTimer.getElapsedMilliseconds() / (data->m_trainingImages.size() + data->m_testImages.size()));

    // Log progress
    std::lock_guard<std::mutex> lock(networksRemainingMutex);
    std::string notification = "Netowork trained! " + std::to_string(--*networksRemaining) + " networks remain.\n";
    std::cout << notification;
}

/*
static void displayImage(const std::vector<double>& imageData)
{
    // This is a quick-and-dirty tool that uses OpenCV to display images to the user
    
    unsigned rows = 28;
    unsigned cols = 28;
    unsigned pixels = rows * cols;
    cv::Mat image(rows, cols, CV_8U);
    uchar* ptr = image.ptr<uchar>(0);
    for (unsigned i = 0; i < pixels; i++) ptr[i] = imageData[i];
    imshow("", image);
    cv::waitKey(0);
    return;
}
*/
