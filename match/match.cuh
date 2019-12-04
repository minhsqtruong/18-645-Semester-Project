#include<iostream>
#include<math.h>
#ifndef MATCH_H
#define MATCH_H


// cpu functions
void cpu_match(int, int, int*, int*, int*);

// gpu functions
// __device__ void gpu_match_Kernel();
__global__ void gpu_match_Loop(int, int, int*, int*, int*, int*);
void gpu_match( int, int, int*, int*, int*, int*);

// test functions
void pipeline_print_match();
#endif