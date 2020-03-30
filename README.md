# NeuralNetwork
This repository contains code from a personal project in the field of neural networks. The code allows users to experiment with training a fully connected neural network to be a classifier for images from the [MNIST database](http://yann.lecun.com/exdb/mnist/). The network has 784 inputs (the number of pixels in each image (28x28)) and 10 outputs (from one-hot encoding each digit from 0 to 9). Users may set the following network hyperparameters:
- number of hidden layers
- number of neurons in each hidden layer
- batch size
- number of epochs 
- learning rate 
- regularization parameter

The user may choose to train multiple networks with the indicated hyperparameters. After the networks are trained and tested the following model metrics are reported:
- maximum accuracy among all models
- mean training time
- mean accuracy
- mean processing time

The code was written using the C++ Standard Library. The only exception is a simple function that displays MNIST images using OpenCV libraries. This has been commented out of the source code for accessibility. The code was compiled with Visual C++ 14.2.

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

-learning_rate lr: a positive number that functions as a multiplier for the weight gradients before the connection weights are updated

(default learning_rate is 0.0001)

-regularization_parameter rp: a non-negative number that functions as a multiplier for the regularization component of the weight gradients before the connection weights are updated

(default regularization_parameter is 0.0)

-n_networks nets: a positive number indicating the number of networks to train with the selected hyperparameters

(default n_networks is 1)


## To-Do
- [ ] Enable GPU computation
- [ ] Enable convolutional layers
