# NeuralNetwork
This repository contains code that allows users to experiment with training a fully connected neural network to be a classifier for images of digits from the [MNIST database](http://yann.lecun.com/exdb/mnist/). The network has 784 inputs (the number of pixels in each 28x28 image) and 10 outputs (from one-hot encoding each digit from 0 to 9). Users may set the following network hyperparameters:
- number of hidden layers
- number of neurons in each hidden layer
- batch size
- number of epochs 
- learning rate 
- regularization parameter

Neural networks with the same hyperparameters may demonstrate different performance after training. This results from the randomization associated with initializing connection weights in the networks and the order in which training data is fed into the networks. Users may select the number of networks to train so they may assess how performance can vary. After the networks are trained and tested the following model metrics are reported:
- maximum accuracy among all models
- mean accuracy
- mean training time
- mean processing time

The code was written using the C++17 standard and compiled with Visual C++ 14.2.

## Usage
### Default:
```
neuralnetwork.exe
```
### Custom:
```
neuralnetwork.exe [-topology n1 [n2…]] [-batch_size b] [-n_epochs e] [-learning_rate lr] [-regularization_parameter rp] [-n_networks nets]
```
-topology n1, n2, etc.: positive integers indicating the number of neurons in each hidden layer of the network

(default topology is 16 16)

-batch_size b: a positive integer indicating the number of training samples to backpropagate before updating the network’s connection weights

(default batch_size is 1)

-n_epochs e: a positive integer indicating the number of times the set of training samples will be backpropagated through the network

(default n_epochs is 1)

-learning_rate lr: a positive number that serves as a multiplier for the weight gradient for each connection weight in the network

(default learning_rate is 0.0002)

-regularization_parameter rp: a non-negative number that serves as a multiplier for the regularization component of the weight gradient for each connection weight in the network

(default regularization_parameter is 0.0)

-n_networks nets: a positive number indicating the number of networks to train with the selected hyperparameters

(default n_networks is 1)

The default network configuration is intended to be a simple demonstration. The parameters were chosen in the interest of a low network training time. Testing has shown an accuracy up to 85% is achievable with this configuration. Higher accuracies may be obtained with experimentation.

Before running the executable, the MNIST files must be extracted and placed in the executable's directory.

## To-Do
- [ ] Enable GPU computation - to improve training times
- [ ] Enable convolutional layers - to improve classification accuracy
