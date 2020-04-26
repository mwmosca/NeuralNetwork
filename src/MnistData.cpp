#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <vector>

#include "MnistData.h"

MnistData::MnistData()
{
    // Extra the data from the MNIST files. Standardize the image data.
    
    // Load MNIST data asynchronously
    std::cout << "Loading MNIST data..." << std::endl;
    std::vector<std::future<bool>> fileFutures;
    fileFutures.reserve(4);
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readLabelFile, this, DataSet::train));
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readImageFile, this, DataSet::train));
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readLabelFile, this, DataSet::test));
    fileFutures.emplace_back(std::async(std::launch::async, &MnistData::readImageFile, this, DataSet::test));

    for (std::future<bool>& f : fileFutures) {
        if(!f.get()) { // Wait for each file to finish reading
            std::cerr << "[ERROR]: MNIST data failed to load." << std::endl;
            return;
        }
    }

    std::cout << "MNIST data loaded!\n" << std::endl
        << "Pre-processing MNIST data..." << std::endl;

    // Calculate statistical measures for the training image data
    if (!calcStatMeasures()) { 
        std::cerr << "[ERROR]: MNIST data pre-processing failed." << std::endl; 
        return;
    }

    // Standardize the image data asynchronously
    std::vector<std::future<void>> stdDataFutures;
    stdDataFutures.reserve(2);
    stdDataFutures.emplace_back(std::async(std::launch::async, &MnistData::standardizeImageData, this, DataSet::train));
    stdDataFutures.emplace_back(std::async(std::launch::async, &MnistData::standardizeImageData, this, DataSet::test));
    
    for (const std::future<void>& f : stdDataFutures) { f.wait(); } // Wait for standardization to complete

    std::cout << "MNIST data pre-processed!\n" << std::endl;

    m_dataValid = true;

    return;
}

bool MnistData::readLabelFile(const DataSet dataSetFlag)
{
    // Open the file containing the desired image labels
    std::string labelFileName = (dataSetFlag == DataSet::train) ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
    std::ifstream inputStream(labelFileName, std::ios::binary);

    if (!inputStream.is_open()) { return false; }
    
    int intReader = 0;            // Temp storage for 32 bit integers read in from the filestream
    unsigned char charReader = 0; // Temp storage for unsigned bytes read in from the filestream
    int n_labels = 0;             // The number of data labels stored in the file
    
    // Read in the magic number. The magic number serves no purpose in this program; 
    // the objective is to advance the filestream's position indicator.
    inputStream.read((char*) &intReader, sizeof(intReader));
    
    inputStream.read((char*) &intReader, sizeof(intReader));    // Read in the number of labels
    n_labels = _byteswap_ulong(intReader);              // Swap from high to low endian for Intel processor compatibility
    
    if (dataSetFlag == DataSet::train) { // Read in the training labels by one-hot encoding digits 0-9
        m_trainingLabels = std::vector<std::vector<double>>(n_labels, std::vector<double>(10));
        for (std::vector<double>& label : m_trainingLabels) {
            inputStream.read((char*) &charReader, sizeof(charReader));
            label[charReader] = 1.0;
        }
    }
    else {                               // Read in the test labels
        m_testLabels = std::vector<double>(n_labels);
        for (double& label : m_testLabels) {
            inputStream.read((char*) &charReader, sizeof(charReader));
            label = charReader;
        }
    }

    inputStream.close();

    return true;
}

bool MnistData::readImageFile(const DataSet dataSetFlag)
{
    // Open the file containing the desired image data
    std::string imageFileName = (dataSetFlag == DataSet::train) ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
    std::ifstream inputStream(imageFileName, std::ios::binary);
    std::vector<std::vector<double>>& imageSet = (dataSetFlag == DataSet::train) ? m_trainingImages : m_testImages;

    if (!inputStream.is_open()) { return false; }

    int intReader = 0;            // Temp storage for 32 bit integers read in from the filestream
    unsigned char charReader = 0; // Temp storage for unsigned bytes read in from the filestream
    int n_images = 0;             // The number of images stored in the file
    int n_rows = 0;               // The number of rows of pixels in each image
    int n_cols = 0;               // The number of columns of pixels in each image
    int n_pixels = 0;             // The number of pixels in each image

    // Read in the magic number. The magic number serves no purpose in this program; 
    // the objective is to advance the filestream's position indicator.
    inputStream.read((char*) &intReader, sizeof(intReader));
    
    inputStream.read((char*) &intReader, sizeof(intReader));    // Read in the number of images
    n_images = _byteswap_ulong(intReader);         // Swap from high to low endian for Intel processor compatibility

    // Read in the image pixel dimensions
    inputStream.read((char*) &intReader, sizeof(intReader));
    n_rows = _byteswap_ulong(intReader);
    inputStream.read((char*) &intReader, sizeof(intReader));
    n_cols = _byteswap_ulong(intReader);
    n_pixels = n_rows * n_cols;

    // Read in image data. Each image is stored as a one-dimensional vector of doubles.
    imageSet = std::vector<std::vector<double>>(n_images, std::vector<double>(n_pixels));
    for (std::vector<double>& image : imageSet) {
        for (double& pixel : image) {
            inputStream.read((char*) &charReader, sizeof(charReader));
            pixel = charReader;
        }
    }

    inputStream.close();
    
    return true;
}

bool MnistData::calcStatMeasures()
{
    // Calculate the statistical measures to standardize the data for each pixel with mean = 0.0 and 
    // standard deviation = 1.0 . Only the training image data are used to calculate these measures.
    
    int n_images = m_trainingImages.size();
    if (!(n_images > 1)) {
        std::cerr << "An insufficient number of MNIST training images were provided to calculate statistical\n" <<
            "measures: " << n_images << " images provided." << std::endl;
        return false;
    }

    int n_pixels = m_trainingImages[0].size();
    m_dataMean = std::vector<double>(n_pixels);
    m_dataStdDev = std::vector<double>(n_pixels);

    // Calculate the mean for each pixel:
    //      mean_p = (1 / m) * (sum i = 1 -> m) x_i
    for (const std::vector<double>& image : m_trainingImages) {
        for (int pixelNum = 0; pixelNum < n_pixels; pixelNum++) {
            m_dataMean[pixelNum] += image[pixelNum];
        }
    }
    for (double& pixelMean : m_dataMean) { 
        pixelMean /= n_images; 
    }

    // Calculate the standard deviation for each pixel:
    //      stdDev_p = sqrt((1 / (m - 1)) * (sum i = 1 -> m) (x_i - mean_p)^2)
    for (const std::vector<double>& image : m_trainingImages) {
        for (int pixelNum = 0; pixelNum < n_pixels; pixelNum++) {
            m_dataStdDev[pixelNum] += std::pow(image[pixelNum] -  m_dataMean[pixelNum], 2);
        }
    }
    for (double& pixelStdDev : m_dataStdDev) {
        // If the standard deviation would be 0.0, instead assign it to 1.0 
        // to avoid division by zero when standardizing the data later
        pixelStdDev = (pixelStdDev == 0.0) ? 1.0 : std::sqrt(pixelStdDev / (n_images - 1));
    }

    return true;
}

void MnistData::standardizeImageData(const DataSet dataSetFlag) 
{
    // Standardize the data for each pixel with mean = 0.0 and standard deviation = 1.0:
    //      s_p = (x_p - mean_p) / stdDev_p
    
    // Standardize the desired image data
    std::vector<std::vector<double>>& imageSet = (dataSetFlag == DataSet::train) ? m_trainingImages : m_testImages;
    int n_pixels = imageSet[0].size();
    for (std::vector<double>& image : imageSet) {
        for (int pixelNum = 0; pixelNum < n_pixels; pixelNum++) {
            image[pixelNum] = (image[pixelNum] - m_dataMean[pixelNum]) / m_dataStdDev[pixelNum];
        }
    }
}
