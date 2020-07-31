#pragma once

#include <vector>

class MnistData
{
    // Description:
    // The purpose of this class is to read in label and image data from the MNIST database.
    // For more information on the MNIST file format visit http://yann.lecun.com/exdb/mnist/ .

    // Design Pattern:  Singleton

  private:
    bool m_dataValid = false;
    std::vector<std::vector<double>> m_trainingLabels;
    std::vector<std::vector<double>> m_trainingImages;
    std::vector<double>              m_testLabels;
    std::vector<std::vector<double>> m_testImages;
    std::vector<double> m_dataMean;
    std::vector<double> m_dataStdDev;

  public:
    ~MnistData() = default;

    MnistData(const MnistData&)            = delete;
    MnistData(MnistData&&)                 = delete;
    MnistData& operator=(const MnistData&) = delete;
    MnistData& operator=(MnistData&&)      = delete;

    // Allows access to a single instance of the class
    static const MnistData& getInstance() { static const MnistData instance; return instance; }

    constexpr bool                                    getDataValid() const { return m_dataValid; }
    constexpr const std::vector<std::vector<double>>& getTrainingLabels() const { return m_trainingLabels; }
    constexpr const std::vector<std::vector<double>>& getTrainingImages() const { return m_trainingImages; }
    constexpr const std::vector<double>&              getTestLabels() const { return m_testLabels; }
    constexpr const std::vector<std::vector<double>>& getTestImages() const { return m_testImages; }

  private:
    MnistData();    // Make constructor inaccessible outside of class
    
    enum class DataSet : char   // Used to differentiate between the training data and test data
    {
        train, test
    };
    
    bool readLabelFile(const DataSet dataSetFlag);
    bool readImageFile(const DataSet dataSetFlag);

    bool calcStatMeasures();
    void standardizeImageData(const DataSet dataSetFlag);
};
