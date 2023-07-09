#ifdef CGRAD_IMPLEMENTATION
    
typedef unsigned char   cgrad_u8;
typedef signed   char   cgrad_s8;
typedef unsigned short  cgrad_u16;
typedef signed   short  cgrad_s16;
typedef unsigned int    cgrad_u32;
typedef signed   int    cgrad_s32;

#ifdef _MSC_VER
  typedef unsigned __int64 cgrad_u64;
  typedef          __int64 cgrad_s64;
#else
  typedef unsigned long long cgrad_u64;
  typedef          long long cgrad_s64;
#endif

#ifdef CGRAD_STATIC
#define CGRAD_DEF static
#else
#define CGRAD_DEF extern
#endif

#ifndef CGRAD_malloc
#include <stdlib.h>
#define CGRAD_malloc(sz)   malloc(sz)
#define CGRAD_free(ptr)    free(ptr)
#endif

#ifndef CGRAD_assert
#include <assert.h>
#define CGRAD_assert(x)    assert(x)
#endif

#ifndef CGRAD_tanh
#include <math.h>
#define CGRAD_tanh(x)      tanhf(x)
#define CGRAD_exp(x)       expf(x)
#define CGRAD_pow(x, y)    powf(x, y)
#define CGRAD_log(x)       logf(x)
CGRAD_DEF float
CGRAD_sigmoid(float A)
{
    float Result = 1.0f/(1.0f + CGRAD_exp(-1.0f*A));
    return(Result);
}
#endif

#ifndef CGRAD_memcpy
#include <string.h>
#define CGRAD_memcpy       memcpy
#define CGRAD_memset       memset
#endif

#ifndef CGRAD_rand
#include <cstdlib>
CGRAD_DEF float
CGRAD_rand(void)
{
    float Result = (float)rand()/(float)RAND_MAX;
    Result *= 2.0f;
    Result -= 1.0f;
    
    return(Result);
}
#endif

enum
{
    CGRAD_OP_CONST,
    CGRAD_OP_WEIGHT,

    CGRAD_OP_ADD,
    CGRAD_OP_SUM,
    CGRAD_OP_MUL,
    CGRAD_OP_EXP,
    CGRAD_OP_POW,
    CGRAD_OP_NEG,
    CGRAD_OP_LOG,
    CGRAD_OP_TANH,
    CGRAD_OP_SIGMOID,
    CGRAD_OP_RELU,

    CGRAD_OP_COUNT,
};

typedef struct cgrad_node
{
    cgrad_u8 Op;
    float Value;
    float Grad;
    cgrad_s64 ParentCount;
    cgrad_node *Parents[0];
} cgrad_node;

typedef struct cgrad_node_ptr_array
{
    cgrad_s64 Max;
    cgrad_s64 Count;
    cgrad_node *Array[0];
} cgrad_node_ptr_array;

typedef struct cgrad_graph
{
    cgrad_s64 NodeCount;
    bool SortedArrayReady;    
    cgrad_node_ptr_array *Sorted;
    cgrad_node *LastNode;
} cgrad_graph;

CGRAD_DEF cgrad_graph *
cgrad_new_graph(void)
{
    cgrad_graph *Result = (cgrad_graph *)CGRAD_malloc(sizeof(cgrad_graph));
    CGRAD_memset(Result, 0, sizeof(cgrad_graph));

    return(Result);
}

CGRAD_DEF cgrad_node_ptr_array *
cgrad_node_array(cgrad_s64 MaxNodes)
{
    cgrad_u64 Size = sizeof(cgrad_node_ptr_array) + sizeof(cgrad_node *)*MaxNodes;
    cgrad_node_ptr_array *Result = (cgrad_node_ptr_array *)CGRAD_malloc(Size);
    CGRAD_memset(Result, 0, Size);
    Result->Max = MaxNodes;
    
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_new_node(cgrad_graph *Graph, cgrad_s64 ParentCount)
{
    cgrad_u64 Size = sizeof(cgrad_node) + sizeof(cgrad_node*)*ParentCount;
    cgrad_node *Result = (cgrad_node *)malloc(Size);
    CGRAD_memset(Result, 0, Size);
    Result->ParentCount = ParentCount;

    Graph->NodeCount++;
    if(ParentCount > 0)
    {
	Graph->LastNode = Result;
	Graph->SortedArrayReady = false;
    }
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_const(cgrad_graph *Graph, float Value)
{
    cgrad_node *Result = cgrad_new_node(Graph, 0);
    Result->Op    = CGRAD_OP_CONST;
    Result->Value = Value;
    return(Result);
};

CGRAD_DEF cgrad_node *
cgrad_weight(cgrad_graph *Graph)
{
    cgrad_node *Result = cgrad_new_node(Graph, 0);
    Result->Op    = CGRAD_OP_WEIGHT;
    Result->Value = CGRAD_rand();

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_mul(cgrad_graph *Graph, cgrad_node *A, cgrad_node *B)
{
    cgrad_node *Result = cgrad_new_node(Graph, 2);
    Result->Op    = CGRAD_OP_MUL;
    Result->Value = A->Value * B->Value;

    Result->Parents[0] = A;
    Result->Parents[1] = B;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_mul_const(cgrad_graph *Graph, cgrad_node *A, float Const)
{
    cgrad_node *ConstTerm = cgrad_const(Graph, Const);    
    cgrad_node *Result = cgrad_mul(Graph, A, ConstTerm);
    
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_neg(cgrad_graph *Graph, cgrad_node *A)
{
    cgrad_node *Result = cgrad_new_node(Graph, 1);
    Result->Op    = CGRAD_OP_NEG;
    Result->Value = -1.0f*A->Value;

    Result->Parents[0] = A;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_add(cgrad_graph *Graph, cgrad_node *A, cgrad_node *B)
{
    cgrad_node *Result = cgrad_new_node(Graph, 2);
    Result->Op    = CGRAD_OP_ADD;
    Result->Value = A->Value + B->Value;

    Result->Parents[0] = A;
    Result->Parents[1] = B;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_add_const(cgrad_graph *Graph, cgrad_node *A, float Const)
{
    cgrad_node *ConstTerm = cgrad_const(Graph, Const);
    cgrad_node *Result = cgrad_add(Graph, A, ConstTerm);

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_sum(cgrad_graph *Graph, cgrad_node_ptr_array *Array)
{
    cgrad_node *Result = cgrad_new_node(Graph, Array->Count);
    Result->Op = CGRAD_OP_SUM;

    float Total = 0.0f;
    for(cgrad_s64 ParentIndex = 0;
	ParentIndex < Array->Count;
	ParentIndex++)
    {
	cgrad_node *Parent = Array->Array[ParentIndex];
	Result->Parents[ParentIndex] = Parent;
	
	Total += Parent->Value;
    }
    Result->Value = Total;
    
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_sub(cgrad_graph *Graph, cgrad_node *A, cgrad_node *B)
{
    cgrad_node *Negative = cgrad_neg(Graph, B);
    cgrad_node *Result = cgrad_add(Graph, A, Negative);
    
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_sub_const(cgrad_graph *Graph, cgrad_node *A, float Const)
{
    cgrad_node *ConstTerm = cgrad_const(Graph, Const);
    cgrad_node *Result = cgrad_sub(Graph, A, ConstTerm);
    
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_exp(cgrad_graph *Graph, cgrad_node *A)
{
    cgrad_node *Result = cgrad_new_node(Graph, 1);
    Result->Op    = CGRAD_OP_EXP;
    Result->Value = CGRAD_exp(A->Value);
    Result->Parents[0] = A;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_pow(cgrad_graph *Graph, cgrad_node *A, cgrad_node *B)
{
    cgrad_node *Result = cgrad_new_node(Graph, 2);
    Result->Op    = CGRAD_OP_POW;
    Result->Value = CGRAD_pow(A->Value, B->Value);
    Result->Parents[0] = A;
    Result->Parents[1] = B;
    
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_pow_const(cgrad_graph *Graph, cgrad_node *A, float B)
{
    cgrad_node *Constant = cgrad_const(Graph, B);
    cgrad_node *Result = cgrad_pow(Graph, A, Constant);
    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_div(cgrad_graph *Graph, cgrad_node *A, cgrad_node *B)
{
    cgrad_node *Recip = cgrad_pow_const(Graph, B, -1.0f);
    cgrad_node *Result = cgrad_mul(Graph, A, Recip);

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_div_const(cgrad_graph *Graph, cgrad_node *A, float B)
{
    cgrad_node *Denominator = cgrad_const(Graph, B);
    cgrad_node *Result = cgrad_div(Graph, A, Denominator);

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_log(cgrad_graph *Graph, cgrad_node *A)
{
    cgrad_node *Result = cgrad_new_node(Graph, 1);
    Result->Op    = CGRAD_OP_LOG;
    Result->Value = CGRAD_log(A->Value);
    Result->Parents[0] = A;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_tanh(cgrad_graph *Graph, cgrad_node *A)
{
    cgrad_node *Result = cgrad_new_node(Graph, 1);
    Result->Op    = CGRAD_OP_TANH;
    Result->Value = CGRAD_tanh(A->Value);
    Result->Parents[0] = A;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_sigmoid(cgrad_graph *Graph, cgrad_node *A)
{
    cgrad_node *Result = cgrad_new_node(Graph, 1);
    Result->Op    = CGRAD_OP_SIGMOID;
    Result->Value = CGRAD_sigmoid(A->Value);
    Result->Parents[0] = A;

    return(Result);
}

CGRAD_DEF cgrad_node *
cgrad_relu(cgrad_graph *Graph, cgrad_node *A)
{
    cgrad_node *Result = cgrad_new_node(Graph, 1);
    Result->Op    = CGRAD_OP_RELU;
    Result->Value = A->Value > 0.0f ? A->Value : 0.0f;
    Result->Parents[0] = A;

    return(Result);
}

CGRAD_DEF bool
cgrad_in_visited(cgrad_node_ptr_array *VisitedArray, cgrad_node *A)
{
    bool Result = false;
    for(cgrad_s64 VisitedIndex = 0;
	VisitedIndex < VisitedArray->Count;
	VisitedIndex++)
    {
	cgrad_node *Visited = VisitedArray->Array[VisitedIndex];
	if(A == Visited)
	{
	    Result = true;
	    break;
	}
    }
    
    return(Result);
}

CGRAD_DEF void
cgrad_topological_sort_(cgrad_node_ptr_array *Sorted, cgrad_node_ptr_array *Visited, cgrad_node *A)
{
    if(!cgrad_in_visited(Visited, A))
    {
	CGRAD_assert(Visited->Count < Visited->Max);
	Visited->Array[Visited->Count++] = A;

	for(cgrad_s64 ParentIndex = 0;
	    ParentIndex < A->ParentCount;
	    ParentIndex++)
	{
	    cgrad_node *Parent = A->Parents[ParentIndex];
	    CGRAD_assert(Parent);
	    cgrad_topological_sort_(Sorted, Visited, Parent);
	}

	CGRAD_assert(Sorted->Count < Sorted->Max);
	Sorted->Array[Sorted->Count++] = A;
    }
}

CGRAD_DEF void
cgrad_topological_sort(cgrad_graph *Graph)
{
    if(!Graph->SortedArrayReady)
    {
	cgrad_node_ptr_array *Visited = cgrad_node_array(Graph->NodeCount);
	Graph->Sorted = cgrad_node_array(Graph->NodeCount);

	cgrad_topological_sort_(Graph->Sorted, Visited, Graph->LastNode);
	CGRAD_free(Visited);
	
	Graph->SortedArrayReady = true;
    }
}

CGRAD_DEF void
cgrad_calculate_values(cgrad_graph *Graph)
{
    cgrad_node_ptr_array *Sorted = Graph->Sorted;
    for(cgrad_s64 SortedIndex = 0;
	SortedIndex < Sorted->Count;
	SortedIndex++)
    {
	cgrad_node *SortedNode = Sorted->Array[SortedIndex];
	
	switch(SortedNode->Op)
	{
	    case CGRAD_OP_CONST:
	    case CGRAD_OP_WEIGHT:
	    {
	    } break;
	    case CGRAD_OP_ADD:
	    {
		CGRAD_assert(SortedNode->ParentCount == 2);
		SortedNode->Value = SortedNode->Parents[0]->Value + SortedNode->Parents[1]->Value;
	    } break;
	    case CGRAD_OP_SUM:
	    {
		float Total = 0.0f;
		for(cgrad_s64 ParentIndex = 0;
		    ParentIndex < SortedNode->ParentCount;
		    ParentIndex++)
		{
		    cgrad_node *Parent = SortedNode->Parents[ParentIndex];	
		    Total += Parent->Value;
		}
		SortedNode->Value = Total;
	    } break;		
	    case CGRAD_OP_MUL:
	    {
		CGRAD_assert(SortedNode->ParentCount == 2);
		SortedNode->Value = SortedNode->Parents[0]->Value * SortedNode->Parents[1]->Value;
	    } break;
	    case CGRAD_OP_EXP:
	    {
		CGRAD_assert(SortedNode->ParentCount == 1);
		SortedNode->Value = CGRAD_exp(SortedNode->Parents[0]->Value);
	    } break;
	    case CGRAD_OP_POW:
	    {
		CGRAD_assert(SortedNode->ParentCount == 2);
		SortedNode->Value = CGRAD_pow(SortedNode->Parents[0]->Value, SortedNode->Parents[1]->Value);
	    } break;
	    case CGRAD_OP_NEG:
	    {
		CGRAD_assert(SortedNode->ParentCount == 1);
		SortedNode->Value = -1.0f*SortedNode->Parents[0]->Value;
	    } break;
	    case CGRAD_OP_LOG:
	    {
		CGRAD_assert(SortedNode->ParentCount == 1);
		SortedNode->Value = CGRAD_log(SortedNode->Parents[0]->Value);
	    } break;
	    case CGRAD_OP_TANH:
	    {
		CGRAD_assert(SortedNode->ParentCount == 1);
		SortedNode->Value = CGRAD_tanh(SortedNode->Parents[0]->Value);
	    } break;
	    case CGRAD_OP_SIGMOID:
	    {
		CGRAD_assert(SortedNode->ParentCount == 1);
		SortedNode->Value = CGRAD_sigmoid(SortedNode->Parents[0]->Value);
	    } break;
	    case CGRAD_OP_RELU:
	    {
		CGRAD_assert(SortedNode->ParentCount == 1);
		float Value = SortedNode->Parents[0]->Value;
		SortedNode->Value = (Value > 0.0f ? Value : 0.0f);
	    } break;
		 
	    CGRAD_assert(0);
	}	
    }
}

CGRAD_DEF void
cgrad_clear_grads(cgrad_graph *Graph)
{
    for(cgrad_s64 NodeIndex = 0;
	NodeIndex < Graph->Sorted->Count;
	NodeIndex++)
    {
	cgrad_node *SortedNode = Graph->Sorted->Array[NodeIndex];
	SortedNode->Grad = 0.0f;
    }    
}

CGRAD_DEF void
cgrad_calculate_grads(cgrad_graph *Graph)
{
    cgrad_node_ptr_array *SortedArray = Graph->Sorted;
    cgrad_node *LastNode = Graph->LastNode;
    
    LastNode->Grad = 1.0f;
    
    for(cgrad_s64 NodeIndex = (SortedArray->Count - 1);
	NodeIndex >= 0;
	NodeIndex--)
    {
	cgrad_node *Sorted = SortedArray->Array[NodeIndex];
	switch(Sorted->Op)
	{
	    case CGRAD_OP_CONST:
	    case CGRAD_OP_WEIGHT:
	    {
	    } break;
	    
	    case CGRAD_OP_ADD:
	    {
		Sorted->Parents[0]->Grad += Sorted->Grad;
		Sorted->Parents[1]->Grad += Sorted->Grad;
	    } break;
	    case CGRAD_OP_SUM:
	    {
		for(cgrad_s64 ParentIndex = 0;
		    ParentIndex < Sorted->ParentCount;
		    ParentIndex++)
		{
		    cgrad_node *Parent = Sorted->Parents[ParentIndex];
		    Parent->Grad += Sorted->Grad;
		}
	    } break;
	    case CGRAD_OP_MUL:
	    {
		Sorted->Parents[0]->Grad += Sorted->Parents[1]->Value * Sorted->Grad;
		Sorted->Parents[1]->Grad += Sorted->Parents[0]->Value * Sorted->Grad;
	    } break;
	    case CGRAD_OP_EXP:
	    {
		Sorted->Parents[0]->Grad += Sorted->Value * Sorted->Grad;
	    } break;
	    case CGRAD_OP_POW:
	    {
		float A = Sorted->Parents[0]->Value;
		float B = Sorted->Parents[1]->Value;

		Sorted->Parents[0]->Grad += B*CGRAD_pow(A, B - 1) * Sorted->Grad;
		Sorted->Parents[1]->Grad += CGRAD_log(A)*CGRAD_pow(A, B) * Sorted->Grad;
	    } break;
	    case CGRAD_OP_NEG:
	    {
		Sorted->Parents[0]->Grad += -1.0f * Sorted->Grad;
	    } break;
	    case CGRAD_OP_LOG:
	    {
		Sorted->Parents[0]->Grad += 1.0f/Sorted->Parents[0]->Value * Sorted->Grad;
	    } break;
	    case CGRAD_OP_TANH:
	    {
		Sorted->Parents[0]->Grad += (1.0f - (Sorted->Value*Sorted->Value)) * Sorted->Grad;
	    } break;
	    case CGRAD_OP_SIGMOID:
	    {
		Sorted->Parents[0]->Grad += (Sorted->Value*(1.0f - Sorted->Value)) * Sorted->Grad;
	    } break;
	    case CGRAD_OP_RELU:
	    {
		Sorted->Parents[0]->Grad += (Sorted->Value <= 0.0f ? 0.0f : 1.0f) * Sorted->Grad;
	    } break;
	    
	    CGRAD_assert(0);
	}
    }    
}

CGRAD_DEF void
cgrad_run_graph(cgrad_graph *Graph)
{
    if(!Graph->SortedArrayReady)
    {
	cgrad_topological_sort(Graph);
    }
    cgrad_calculate_values(Graph);
    cgrad_clear_grads(Graph);
    cgrad_calculate_grads(Graph);
}

#endif //CGRAD_IMPLEMENTATION
