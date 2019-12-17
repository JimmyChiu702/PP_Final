#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

#include "MLP_OutputLayer.h"

class  MLP_HiddenLayer {
    int nPreviousUnit = 784;
    int nCurrentUnit = 512;
    
    float* inputLayer;
    float outputLayer[512];
    float weight[410000];
    float gradient[410000];
    float delta[512];
    
    float biasWeight[512];    
    float biasGradient[512];
    
public:
    MLP_HiddenLayer(){
        int seed = 1;
        srand(seed);
        for (int j = 0; j < nCurrentUnit; j++)
        {
            outputLayer[j]=0.0;
            delta[j]=0.0;
            for (int i = 0; i < nPreviousUnit; i++)
            {
                weight[j*nPreviousUnit+i]   = 0.2 * rand() / RAND_MAX - 0.1;
                gradient[j*nPreviousUnit+i]= 0.0;
            }
            biasWeight[j] = 0.2 * rand() / RAND_MAX - 0.1;                             
            biasGradient[j] = 0;
        }
    }
    ~MLP_HiddenLayer(){};
    
    float* ForwardPropagate(float* inputLayer);
    void BackwardPropagateHiddenLayer(MLP_OutputLayer* previousLayer);
    
    void UpdateWeight(float learningRate);
    
	float* GetOutput()  {   return outputLayer; }
    float* GetWeight()  {   return weight;      }
    float* GetDelta()   {   return delta;       }
    int GetNumCurrent() {   return nCurrentUnit;}
	int GetMaxOutputIndex();
    // Sigmoid
    float ActivationFunction(float net)		{ return 1.F/(1.F + (float)exp(-net)); }
    //float DerivativeActivation(int preNode){return (1 - outputLayer[preNode]) *outputLayer[preNode]; }
    
    // ReLU
    //float ActivationFunction(float net)  {if (net <= 0) net=0; return net;}
    //float DerivativeActivation(int preNode){ if(outputLayer[preNode] <= 0) return 0.01; else return 1;}
    
    
    float DerActivationFromOutput(float output){ return output * (1.F-output); }
    float DerActivation(float net)	{ return DerActivationFromOutput(ActivationFunction(net)); }
    
    
    void test() {
        for (int i=0; i<50; i++) {
            std::cout << gradient[i] << " " << delta[i] << " " << biasGradient[i] << " " << outputLayer[i] << " " << weight[i] << std::endl;
        }
    }
};