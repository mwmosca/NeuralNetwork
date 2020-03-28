# NeuralNetwork
This repository contains code from a personal project in the field of neural networks. The code allows users to experiment with training a fully connected neural network to act as a classifier for images from the MNIST database (http://yann.lecun.com/exdb/mnist/). Users may set network hyperparameters including the number of hidden layers, the number of neurons in each hidden layer, the batch size, the number of epochs, the learning rate, and the regularization parameter. After the network is trained and tested the training time, accuracy, and processing time of the model are reported.

The code was written using the C++ Standard Library and compiled with Visual C++ 14.2. The only exception is a simple function that displays MNIST images using OpenCV libraries. This has been commented out of the source code for accessibility.

## To-Do
* Enable GPU computation
* Enable convolutional layers
