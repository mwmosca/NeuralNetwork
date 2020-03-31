#pragma once

#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <vector>

#include "MnistData.h"

// The purpose of this class is to read in label and image data from the MNIST database. For more information on the MNIST
// file format visit http://yann.lecun.com/exdb/mnist/ .

MnistData::MnistData() : m_dataValid(false)
{
    // Holds the std::future<bool> return values from std::async
    std::vector<std::future<bool>> fileFutures;

    // 3 files will be read asynchronously:
    //      1) the training labels file
    //      2) the training images file
    //      3) the test labels file
    fileFutures.reserve(3);
    
    std::cout << "Loading and pre-processing MNIST data..." << std::endl;
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readMnistTrainLabelFile, this));
    // The mean and standard deviation of the training image data are calculated as the data is read. These measures
    // are then used to standardize the data before it is passed into the neural network.
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readMnistTrainImageFile, this));
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readMnistTestLabelFile, this));
    for (std::future<bool>& i : fileFutures) i.wait();      // Wait until the three files have been read completely
    
    // The training image data must be completely read in before reading in the test image data because the mean and standard
    // deviation of the training data will also be used to standardize the test image data.
    bool overallSatus = readMnistTestImageFile();

    // Check if all files were read properly
    for (std::future<bool>& i : fileFutures) overallSatus = overallSatus && i.get();
    m_dataValid = overallSatus;
    if (m_dataValid) std::cout << "MNIST data loaded and pre-processed!" << std::endl;
    else std::cerr << "MNIST data failed to load." << std::endl;
}

bool MnistData::readMnistTrainLabelFile()
{
    bool status = false;    // Set up a normally open status flag

    // Open the file containing the labels for the training images
    std::ifstream inputStream("train-labels.idx1-ubyte", std::ios::binary);

    if (inputStream.is_open()) {
        unsigned intReader = 0;         // Temp storage for ints read in from the filestream
        unsigned char charReader = 0;   // Temp storage for chars read in from the filestream
        unsigned numSamples = 0;        // The number of training data labels stored in the file
        
        // Read in the magic number. The magic number serves no purpose in this program. The objective is to advance
        // the filestream's output position indicator.
        inputStream.read((char*) &intReader, sizeof(intReader));
        
        // Read in the number of samples
        inputStream.read((char*) &intReader, sizeof(intReader));
        // Swap from high to low endian for Intel processor compatibility
        numSamples = _byteswap_ulong(intReader);
        
        // Read in the training labels using one-hot encoding
        m_trainingLabels.reserve(numSamples);
        for (unsigned labelNum = 0; labelNum < numSamples; labelNum++) {
            inputStream.read((char*) &charReader, sizeof(charReader));
            m_trainingLabels.emplace_back(std::vector<double> {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            m_trainingLabels.back()[charReader] = 1.0;
        }

        inputStream.close();
        status = true;
    }

    return status;
}

bool MnistData::readMnistTrainImageFile()
{
    bool status = false;    // Set up a normally open status flag
    
    // Open the file containing the training image data
    std::ifstream inputStream("train-images.idx3-ubyte", std::ios::binary);

    if (inputStream.is_open()) {
        unsigned intReader = 0;         // Temp storage for ints read in from the filestream
        unsigned char charReader = 0;   // Temp storage for chars read in from the filestream
        unsigned numSamples = 0;        // The number of training images stored in the file
        unsigned rows = 0;              // The number of rows of pixels in each image
        unsigned cols = 0;              // The number of columns of pixels in each image
        unsigned pixels = 0;            // The number of pixels in each image

        // Read in the magic number. The magic number serves no purpose in this program. The objective is to advance
        // the filestream's output position indicator.
        inputStream.read((char*) &intReader, sizeof(intReader));

        // Read in the number of training image samples
        inputStream.read((char*) &intReader, sizeof(intReader));
        // Swap from high to low endian for Intel processor compatibility
        numSamples = _byteswap_ulong(intReader);

        // Read in the image pixel dimensions
        inputStream.read((char*) &intReader, sizeof(intReader));
        rows = _byteswap_ulong(intReader);
        inputStream.read((char*) &intReader, sizeof(intReader));
        cols = _byteswap_ulong(intReader);
        pixels= rows * cols;

        // Read in image data and begin collecting mean data for each pixel
        m_trainingImages.reserve(numSamples);
        m_dataMean.reserve(pixels);
        m_dataMean = std::vector<double>(pixels, 0.0);
        for (unsigned imageNum = 0; imageNum < numSamples; imageNum++) {
            std::vector<double> imageData;      // Each image is stored as a one-dimensional vector of doubles
            imageData.reserve(pixels);
            for (unsigned pixelNum = 0; pixelNum < pixels; pixelNum++) {
                inputStream.read((char*) &charReader, sizeof(charReader));
                imageData.emplace_back(charReader);
                m_dataMean[pixelNum] += charReader;     // Mean summation step: meanSum_p = (sum i = 1 -> m) x_i
            }
            m_trainingImages.emplace_back(imageData);
        }

        inputStream.close();
        
        // Standardize the data for each pixel: mean = 0.0, standard deviation = 1.0
        // mean_p = (1 / m) * (sum i = 1 -> m) x_i
        // stdDev_p = sqrt((1 / (m - 1)) * (sum i = 1 -> m) (x_i - mean_p)^2)
        // s_p = (x_p - mean_p) / stdDev_p
        
        // Mean division step:
        //      mean_p = (1 / m) * meanSum_p
        for (unsigned pixelNum = 0; pixelNum < m_dataMean.size(); pixelNum++) m_dataMean[pixelNum] /= numSamples;

        // Set up standard deviation data for each pixel
        m_dataStdDev.reserve(pixels);
        m_dataStdDev = std::vector<double>(pixels, 0.0);
        
        // Standard deviation summation step:
        //      stdDevSum_p = (sum i = 1 -> m) (x_i - mean_p)^2
        for (unsigned imageNum = 0; imageNum < numSamples; imageNum++) {
            for (unsigned pixelNum = 0; pixelNum < pixels; pixelNum++) {
                m_dataStdDev[pixelNum] += std::pow((m_trainingImages[imageNum][pixelNum] - m_dataMean[pixelNum]), 2);
            }
        }

        // Standard deviation division and square root step: 
        //      stdDev_i = sqrt((1 / (m - 1)) * stdDevSum_p)
        for (unsigned pixelNum = 0; pixelNum < m_dataStdDev.size(); pixelNum++) {
            // If the standard deviation would be 0.0 instead assign it to 1.0 to avoid division by zero when 
            // standardizing the data
            m_dataStdDev[pixelNum] = (m_dataStdDev[pixelNum] == 0.0) ? 
                1.0 : std::sqrt(m_dataStdDev[pixelNum] / (numSamples - 1));
        }

        // Standardize the training data:
        //      s_p = (x_p - mean_p) / stdDev_p
        for (unsigned imageNum = 0; imageNum < numSamples; imageNum++) {
            for (unsigned pixelNum = 0; pixelNum < pixels; pixelNum++) {
                m_trainingImages[imageNum][pixelNum] = 
                    (m_trainingImages[imageNum][pixelNum] - m_dataMean[pixelNum]) / m_dataStdDev[pixelNum];
            }
        }

        status = true;
    }
    
    return status;
}

bool MnistData::readMnistTestLabelFile() 
{
    bool status = false;    // Set up a normally open status flag
    
    // Open the file containing the labels for the test images
    std::ifstream inputStream("t10k-labels.idx1-ubyte", std::ios::binary);

    if (inputStream.is_open()) {
        unsigned intReader = 0;         // Temp storage for ints read in from the filestream
        unsigned char charReader = 0;   // Temp storage for chars read in from the filestream
        unsigned numSamples = 0;        // The number of test data labels stored in the file
        
        // Read in the magic number. The magic number serves no purpose in this program. The objective is to advance
        // the filestream's output position indicator.
        inputStream.read((char*) &intReader, sizeof(intReader));
        
        // Read in the number of samples
        inputStream.read((char*) &intReader, sizeof(intReader));
        // Swap from high to low endian for Intel processor compatibility
        numSamples = _byteswap_ulong(intReader);
        
        // Read in the test labels
        m_testLabels.reserve(numSamples);
        for (unsigned i = 0; i < numSamples; i++) {
            inputStream.read((char*) &charReader, sizeof(charReader));
            m_testLabels.emplace_back(charReader);
        }

        inputStream.close();
        status = true;
    }

    return status;
}

bool MnistData::readMnistTestImageFile() 
{
    bool status = false;    // Set up a normally open status flag
    
    // Open the file containing the test image data
    std::ifstream inputStream("t10k-images.idx3-ubyte", std::ios::binary);

    if (inputStream.is_open()) {
        unsigned intReader = 0;         // Temp storage for ints read in from the filestream
        unsigned char charReader = 0;   // Temp storage for chars read in from the filestream
        unsigned numSamples = 0;        // The number of test images stored in the file
        unsigned rows = 0;              // The number of rows of pixels in each image
        unsigned cols = 0;              // The number of columns of pixels in each image
        unsigned pixels = 0;            // The number of pixels in each image

        // Read in the magic number. The magic number serves no purpose in this program. The objective is to advance
        // the filestream's output position indicator.
        inputStream.read((char*) &intReader, sizeof(intReader));

        // Read in the number of test image samples
        inputStream.read((char*) &intReader, sizeof(intReader));
        // Swap from high to low endian for Intel processor compatibility
        numSamples = _byteswap_ulong(intReader);

        // Read in the image pixel dimensions
        inputStream.read((char*) &intReader, sizeof(intReader));
        rows = _byteswap_ulong(intReader);
        inputStream.read((char*) &intReader, sizeof(intReader));
        cols = _byteswap_ulong(intReader);
        pixels = rows * cols;

        // Read in image data
        m_testImages.reserve(numSamples);
        for (unsigned imageNum = 0; imageNum < numSamples; imageNum++) {
            std::vector<double> imageData;
            imageData.reserve(pixels);
            for (unsigned pixelNum = 0; pixelNum < pixels; pixelNum++) {
                inputStream.read((char*) &charReader, sizeof(charReader));
                // Standardize the data for each pixel using the measures from the training image data: 
                //      s_p = (x_p - mean_p) / stdDev_p
                imageData.emplace_back((charReader - m_dataMean[pixelNum]) / m_dataStdDev[pixelNum]);
            }
            m_testImages.emplace_back(imageData);
        }

        inputStream.close();
        status = true;
    }

    return status;
}
