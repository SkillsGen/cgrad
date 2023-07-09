#include "stdio.h"

#define CGRAD_IMPLEMENTATION 1
#include "cgrad.h"

float _OrGate[4][3]
{
    { 0, 0,   0},
    { 1, 0,   1},
    { 0, 1,   1},
    { 1, 1,   1},
};

float _AndGate[4][3]
{
    { 0, 0,   0},
    { 1, 0,   0},
    { 0, 1,   0},
    { 1, 1,   1},
};

float _XorGate[4][3]
{
    { 0, 0,   0},
    { 1, 0,   1},
    { 0, 1,   1},
    { 1, 1,   0},
};

#define DATA _OrGate

int main(void)
{
    cgrad_graph *NNGraph = cgrad_new_graph();


    //
    //NOTE: Single Neuron:
    cgrad_node *Weight1 = cgrad_weight(NNGraph);
    cgrad_node *Weight2 = cgrad_weight(NNGraph);
    cgrad_node *Bias    = cgrad_weight(NNGraph);

    
    cgrad_node *TotalLoss = cgrad_const(NNGraph, 0.0f);
    for(int Row = 0;
	Row < 4;
	Row++)
    {
	cgrad_node *Input1 = cgrad_const(NNGraph, DATA[Row][0]);
	cgrad_node *Input2 = cgrad_const(NNGraph, DATA[Row][1]);

	cgrad_node *Input1Weight1 = cgrad_mul(NNGraph, Input1, Weight1);
	cgrad_node *Input2Weight2 = cgrad_mul(NNGraph, Input2, Weight2);

	cgrad_node *Sum1 = cgrad_add(NNGraph, Input1Weight1, Input2Weight2);
	cgrad_node *Sum2 = cgrad_add(NNGraph, Sum1, Bias);

	//NOTE: Activation function
	cgrad_node *NeuronOutput = cgrad_sigmoid(NNGraph, Sum2);

	
	//
	//NOTE: Loss	
	cgrad_node *CorrectOutput = cgrad_const(NNGraph, DATA[Row][2]);

	cgrad_node *Difference   = cgrad_sub(NNGraph, NeuronOutput, CorrectOutput);
	cgrad_node *DifferenceSq = cgrad_pow_const(NNGraph, Difference, 2.0f);

	TotalLoss = cgrad_add(NNGraph, TotalLoss, DifferenceSq);
    }

    cgrad_node *Loss = cgrad_div_const(NNGraph, TotalLoss, 4.0f);

    cgrad_run_graph(NNGraph);


    //
    //NOTE: Training
    float LearningRate = 1.0f;    
    for(int Epoch = 0;
	Epoch < 100;
	Epoch++)
    {
	for(cgrad_s64 NodeIndex = 0;
	    NodeIndex < NNGraph->Sorted->Count;
	    NodeIndex++)
	{
	    cgrad_node *Node = NNGraph->Sorted->Array[NodeIndex];
	    if(Node->Op == CGRAD_OP_WEIGHT)
	    {
		Node->Value -= LearningRate*Node->Grad;
//		printf("Value: %f, Grad: %f\n", Node->Value, Node->Grad);
	    }
	}

	cgrad_run_graph(NNGraph);
	printf("Loss: %f\n", Loss->Value);
    }


    //
    //NOTE: Validation
    for(int Row = 0;
	Row < 4;
	Row++)
    {
	float Input1 = DATA[Row][0];
	float Input2 = DATA[Row][1];
	
	float NeuronSum = Input1*Weight1->Value + Input2*Weight2->Value + Bias->Value;
	float Activated = CGRAD_sigmoid(NeuronSum);

	float CorrectOutput = DATA[Row][2];

	printf("%f , %f    NN: %f  Correct:%f\n", Input1, Input2, Activated, CorrectOutput);
    }
    
    return(1);
}
