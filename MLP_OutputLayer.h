#ifndef _OUTPUTLAYER_H_
#define _OUTPUTLAYER_H_

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

class  MLP_OutputLayer {
    int nPreviousUnit = 512;
    int nCurrentUnit = 10;
    
    float* inputLayer;
    float outputLayer[10];
    float weight[5120];
    float gradient[5120];
    float delta[10];
    
    float biasWeight[10];    
    float biasGradient[10];
    
public:
    MLP_OutputLayer(){
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
    };
    ~MLP_OutputLayer(){};
    
    float* ForwardPropagate(float* inputLayer);
    void BackwardPropagateOutputLayer(float* desiredValues);
    
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
        for (int i=0; i<10; i++) {
            std::cout << gradient[i] << " " << delta[i] << " " << biasGradient[i] << " " << outputLayer[i] << " " << weight[i] << std::endl;
        }
    }
};

#endif