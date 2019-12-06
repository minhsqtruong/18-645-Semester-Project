/*
Benchmark GPU on the following operation:
Arithmetic: + - * / %
Single Precision Intrinsics: __cosf __sinf __fadd_rd __fdiv_rd __fmad_rd __fmul_rd
Integer Intrinsics: __hadd __mul24 __sad
*/

#include<cuda.h>
#include<stdio.h>
/*=============================================================
		INTEGER ARITHMETIC LATENCY KERNEL
=============================================================*/
__global__
void test(int * globvar)
{
	int  var = globvar[0];
	//printf("JERE\n");
	for (int i = 0 ; i < 1000000000; i++) {
		//printf("%d\n", i);
		var = var << 1;
	}
	globvar[0] = var;
	printf("%d\n", var);
}

int main(void)
{
	int * globvar;
	cudaMallocManaged(&globvar, sizeof(int) * 1);
	globvar[0] = 1;
	test<<<1,1>>>(globvar);
	cudaDeviceSynchronize();
	//printf("%d\n", globvar);
}
