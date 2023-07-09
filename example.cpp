#include "stdio.h"

#define CGRAD_IMPLEMENTATION 1
#include "cgrad.h"


int main(void)
{
    cgrad_graph *Graph = cgrad_new_graph();
    cgrad_node *A = cgrad_const(Graph, 6.0f);
    cgrad_node *B = cgrad_const(Graph, -3.0f);

    cgrad_node *C = cgrad_add(Graph, A, B);
    cgrad_node *D = cgrad_pow_const(Graph, B, 3.0f);
    cgrad_node *E = cgrad_div(Graph, C, D);

    cgrad_run_graph(Graph);
    printf("%f\n", E->Value);
    printf("%f\n", A->Grad);
    A->Value -= 1.0f;
    cgrad_run_graph(Graph);
    printf("%f\n", E->Value);

    return(1);
}
    
