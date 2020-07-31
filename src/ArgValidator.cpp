#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>

#include "ArgValidator.h"

void ArgValidator::initInstance(const int argc, const char* const * const argv)
{   
    // Initializes the singleton instance with custom hyperparameters. If getInstance() is 
    // called without calling initInstance() first the default hyperparameters will be used.
    
    ArgValidator::getInstanceImpl(argc, argv);
}

const ArgValidator& ArgValidator::getInstance()
{
    // Provides acces to the singleton instance.
    
    return getInstanceImpl();
}

const ArgValidator& ArgValidator::getInstanceImpl(const int argc, const char* const * const argv)
{
    // Retrieves the singleton instance. The instance may be initialized with initInstance() for custom hyperparameter 
    // configurations. Default arguments are used so that the function may be called without parameters once the instance 
    // has been initialized.
    
    static const ArgValidator instance(argc, argv); 
    return instance;
}

ArgValidator::ArgValidator(const int argc, const char* const * const argv)
    : m_argc(argc), m_argv(argv)
{
    // Keeps track of arguments that have been processed to catch repeated argument
    std::set<std::string> processedArgs;

    // Parse command line options:
    for (int i = 1; i < m_argc; i++) 
    {    
        if (processedArgs.find(m_argv[i]) != processedArgs.end()) {
            m_argErrors << "[ERROR]: Argument index " << i << ", value " << m_argv[i] << ": argument " << 
                m_argv[i] << " has already been processed.\n";
        }

        else if (!std::strcmp(m_argv[i], "-topology")) { // Read in the network topology
            processedArgs.emplace(m_argv[i]);
            
            // Delete the default topology but keep the input layer
            m_topology.erase(m_topology.begin() + 1, m_topology.end());

            // Read in topology arguments until all the numeric arguments 
            // have been processed or a non-numeric argument is reached
            double argVal;
            for (; i + 1 < m_argc && ArgValidator::isNumeric(argv[i + 1], argVal); i++) {
                if (argVal > 0 && ArgValidator::isInteger(argVal)) {
                    m_topology.push_back(argVal);
                }
                else {
                    m_argErrors << "[ERROR]: Argument index " << i + 1 << ", value " << argVal << 
                        ": -topology arguments must be a positive integers.\n";
                }
            }

            m_topology.push_back(10);       // Add the output layer to the topology

            if (m_topology.size() < 3) {    // The topology must have at least 1 hidden layer
                m_argErrors << "[ERROR]: No valid arguments were provided for -topology.\n";
            }
        }

        else if (!std::strcmp(m_argv[i], "-batch_size")) {                // Read in the batch size
            processedArgs.emplace(m_argv[i]);
            getHyperparameterPosInt(i, m_batchSize);
        }
        else if (!std::strcmp(m_argv[i], "-n_epochs")) {                  // Read in the number of epochs
            processedArgs.emplace(m_argv[i]);
            getHyperparameterPosInt(i, m_epochs);
        }
        else if (!std::strcmp(m_argv[i], "-learning_rate")) {             // Read in the learning rate
            processedArgs.emplace(m_argv[i]);
            getHyperparameterPosReal(i, m_alpha);
        }
        else if (!std::strcmp(m_argv[i], "-regularization_parameter")) {  // Read in the regularization parameter
            processedArgs.emplace(m_argv[i]);
            getHyperparameterNonnegReal(i, m_lambda);
        }
        else if (!std::strcmp(m_argv[i], "-n_networks")) {                // Read in the number of networks
            processedArgs.emplace(m_argv[i]);
            getHyperparameterPosInt(i, m_numNetworks);
        }
        else {
            m_argErrors << "[ERROR]: Argument index " << i << ", value " << m_argv[i] << ": argument not recognized.\n";
        }
    }
    
    return;
}

bool ArgValidator::isNumeric(const char* const strSource, double& valResult)
{
    // Determines if a character array may be converted to a numeric type. 
    // A valid conversion is assigned to the double passed by reference.

    // If the character array can be successfully streamed into a double, it is numeric.
    std::stringstream sourceReader;
    double convertedVal;
    sourceReader << strSource;
    sourceReader >> convertedVal;

    if (sourceReader) {
        valResult = convertedVal;
        return true;
    }

    return false;
}

bool ArgValidator::isInteger(const double val)
{
    // Determines if val is an integer.
    
    return std::nearbyint(val) == val;
}

bool ArgValidator::nextArgIsNumeric(const int argIndex, double& argVal)
{
    // Determines if the next argument is present and numeric. If so, 
    // the argument will be assigned to the double passed by reference.
    
    if (argIndex + 1 >= m_argc) {
        m_argErrors << "[ERROR]: End of argument list reached. No argument was provided for the value of " << 
            m_argv[argIndex] << ".\n";
        return false;
    }

    if (!ArgValidator::isNumeric(m_argv[argIndex + 1], argVal)) {
        m_argErrors << "[ERROR]: Argument index " << argIndex + 1 << ", value " << m_argv[argIndex + 1] << ": the " << 
            m_argv[argIndex] << " argument must be numeric.\n";
        return false;
    }
    
    return true;
}

void ArgValidator::getHyperparameterPosInt(int& hyperparameterIndex, int& hyperparameter)
{
    // Determines if the next available argument is a positive integer. If so, it is assigned to the int passed by reference.

    double argVal;
    if (nextArgIsNumeric(hyperparameterIndex, argVal)) {
        hyperparameterIndex++;
        
        if (argVal > 0 && ArgValidator::isInteger(argVal)) {
            hyperparameter = argVal;
        }
        else {
            m_argErrors << "[ERROR]: Argument index " << hyperparameterIndex << ", value " << argVal << ": the " << 
                m_argv[hyperparameterIndex - 1] << " argument must be a positive integer.\n";
        }
    }

    return;
}

void ArgValidator::getHyperparameterPosReal(int& hyperparameterIndex, double& hyperparameter)
{
    // Determines if the next available argument is a positive real 
    // number. If so, it is assigned to the double passed by reference.

    double argVal;
    if (nextArgIsNumeric(hyperparameterIndex, argVal)) {
        hyperparameterIndex++;
        
        if (argVal > 0) {
            hyperparameter = argVal;
        }
        else {
            m_argErrors << "[ERROR]: Argument index " << hyperparameterIndex << ", value " << argVal << ": the " << 
                m_argv[hyperparameterIndex - 1] << " argument must be positive.\n";
        }
    }

    return;
}

void ArgValidator::getHyperparameterNonnegReal(int& hyperparameterIndex, double& hyperparameter)
{
    // Determines if the next available argument is a nonnegative real 
    // number. If so, it is assigned to the double passed by reference.

    double argVal;
    if (nextArgIsNumeric(hyperparameterIndex, argVal)) {
        hyperparameterIndex++;
        
        if (argVal >= 0) {
            hyperparameter = argVal;
        }
        else {
            m_argErrors << "[ERROR]: Argument index " << hyperparameterIndex << ", value " << argVal << ": the " << 
                m_argv[hyperparameterIndex - 1] << " argument must be nonnegative.\n";
        }
    }

    return;
}
