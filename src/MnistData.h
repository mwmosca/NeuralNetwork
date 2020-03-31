#pragma once

#include <vector>

class MnistData
{
    public:
        MnistData();
        std::vector<std::vector<double>> m_trainingLabels;
        std::vector<std::vector<double>> m_trainingImages;
        std::vector<double> m_testLabels;
        std::vector<std::vector<double>> m_testImages;
        bool getDataValid() { return m_dataValid; }
    private:
        bool m_dataValid;
        std::vector<double> m_dataMean;
        std::vector<double> m_dataStdDev;
        bool readMnistTrainLabelFile();
        bool readMnistTrainImageFile();
        bool readMnistTestLabelFile();
        bool readMnistTestImageFile();
};
