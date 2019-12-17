#include <omp.h>

#include "MLP_HiddenLayer.h"

float* MLP_HiddenLayer::ForwardPropagate(float* inputLayers)      // f( sigma(weights * inputs) + bias )
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

void MLP_HiddenLayer::BackwardPropagateHiddenLayer(MLP_OutputLayer* previousLayer)
{
    
    float* previousLayer_weight = previousLayer->GetWeight();
    float* previousLayer_delta = previousLayer->GetDelta();
    int previousLayer_node_num = previousLayer->GetNumCurrent();

    for (int j = 0; j < nCurrentUnit; j++)
    {
        float previous_sum=0;
        for (int k = 0; k < previousLayer_node_num; k++)
        {
            previous_sum += previousLayer_delta[k] * previousLayer_weight[k*nCurrentUnit + j];
        }
        delta[j] =  outputLayer[j] * (1 - outputLayer[j])* previous_sum;
        //delta[j] =  DerivativeActivation(j)* previous_sum;
    }
    
    for (int j = 0; j < nCurrentUnit; j++)
        for (int i = 0; i < nPreviousUnit ; i++)
            gradient[j*nPreviousUnit + i] +=  -delta[j] * inputLayer[i];
    
    for (int j = 0 ; j < nCurrentUnit   ; j++)
        biasGradient[j] += -delta[j] ;
}

void MLP_HiddenLayer::UpdateWeight(float learningRate)
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


int MLP_HiddenLayer::GetMaxOutputIndex()
{
    int maxIdx = 0;
    for(int o = 1; o < nCurrentUnit; o++){
        if(outputLayer[o] > outputLayer[maxIdx])
            maxIdx = o;
    }
    
    return maxIdx;
}


