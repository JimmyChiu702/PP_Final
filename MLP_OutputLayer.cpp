#include <omp.h>

#include "MLP_OutputLayer.h"

float* MLP_OutputLayer::ForwardPropagate(float* inputLayers)      // f( sigma(weights * inputs) + bias )
{
    this->inputLayer=inputLayers;

    for(int j = 0 ; j < nCurrentUnit ; j++)
    {
        float net= 0;
        for(int i = 0 ; i < nPreviousUnit ; i++)
        {
            net += inputLayer[i] * weight[j*nPreviousUnit+i];
        }
        net+=biasWeight[j];
        
        outputLayer[j] = ActivationFunction(net);
    }
    return outputLayer;
}

void MLP_OutputLayer::BackwardPropagateOutputLayer(float* desiredValues)
{
    for (int k = 0; k < nCurrentUnit; k++){
        float fnet_derivative = outputLayer[k] * (1 - outputLayer[k]);
        delta[k] = fnet_derivative * (desiredValues[k] - outputLayer[k]);
        //delta[k] = DerivativeActivation(k) * (desiredValues[k] - outputLayer[k]);
    }
    
    for (int k = 0 ; k < nCurrentUnit ; k++)
        for (int j = 0 ; j < nPreviousUnit; j++)
            gradient[k*nPreviousUnit + j] += - (delta[k] * inputLayer[j]);
    
    for (int k = 0 ; k < nCurrentUnit   ; k++)
            biasGradient[k] += - delta[k] ;
    
    
}

void MLP_OutputLayer::UpdateWeight(float learningRate)
{
    for (int j = 0; j < nCurrentUnit; j++)
        for (int i = 0; i < nPreviousUnit; i++)
            weight[j*nPreviousUnit + i] +=  -learningRate *gradient[j*nPreviousUnit + i];
    
    for (int j = 0; j < nCurrentUnit; j++)
        biasWeight[j] += -biasGradient[j];
    
    for (int j = 0; j < nCurrentUnit; j++)           
        for (int i = 0; i < nPreviousUnit; i++)
            gradient[j*nPreviousUnit + i] = 0;
    
    for (int j = 0; j < nCurrentUnit; j++)
        biasGradient[j]=0;
}


int MLP_OutputLayer::GetMaxOutputIndex()
{
    int maxIdx = 0;
    for(int o = 1; o < nCurrentUnit; o++){
        if(outputLayer[o] > outputLayer[maxIdx])
            maxIdx = o;
    }
    
    return maxIdx;
}


