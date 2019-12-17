#ifndef MLP_Network_H
#define MLP_Network_H
#include "MLP_HiddenLayer.h"
#include "MLP_OutputLayer.h"


#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

class MLP_Network {
    
private:
    int nInputUnit = 784;
    int nHiddenUnit = 512;
    int nOutputUnit = 10;
    int nHiddenLayer = 1;

    MLP_HiddenLayer HiddenLayer;
    MLP_OutputLayer OutputLayer;
public:
    MLP_Network(){}
    ~MLP_Network(){}
    
    void Train();
    void ForwardPropagateNetwork(float* inputNetwork);
    void BackwardPropagateNetwork(float* desiredOutput);
    void UpdateWeight(float learningRate);
    float CostFunction(float* inputNetwork,float* desiredOutput);
    
    float CalculateResult(float* inputNetwork,float* desiredOutput);

    void test() {
        HiddenLayer.test();
    }
};

#endif
