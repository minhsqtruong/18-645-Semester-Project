/*
Benchmark GPU on the following operation:
Arithmetic: + - * / % 
Single Precision Intrinsics: __cosf __sinf __fadd_rd __fdiv_rd __fmad_rd __fmul_rd
Integer Intrinsics: __hadd __mul24 __sad 
*/

#define STREAM_LENGTH 1000
#include<cuda.h>

/*=============================================================
		INTEGER ARITHMETIC LATENCY KERNEL
=============================================================*/
__global__ 
void i_add_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 0;
	int a = 1;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c+=a;
	end = clock64();
	printf("Integer add latency: %llu\n", end - start);
	
}

__global__ 
void i_min_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 0;
	int a = 1;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c-=a;
	end = clock64();
	printf("Integer minus latency: %llu\n", end - start);
	
}

__global__ 
void i_mul_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 1;
	int a = 1;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c*=a;
	end = clock64();
	printf("Integer multiply latency: %llu\n", end - start);
	
}

__global__ 
void i_mod_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 0;
	int a = 1;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c%=a;
	end = clock64();
	printf("Integer modulo latency: %llu\n", end - start);
	
}

/*=============================================================
	    SINGLE PRECISION ARITHMETIC LATENCY KERNEL
=============================================================*/
__global__ 
void f_add_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 0.0;
	float a = 1.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c+=a;
	end = clock64();
	printf("Single Precision add latency: %llu\n", end - start);
	
}

__global__ 
void f_min_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 0.0;
	float a = 1.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c-=a;
	end = clock64();
	printf("Single Precision minus latency: %llu\n", end - start);
	
}

__global__ 
void f_mul_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	float a = 1.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c*=a;
	end = clock64();
	printf("Single Precision multiply latency: %llu\n", end - start);
	
}

__global__ 
void f_div_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	float a = 2.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c/=a;
	end = clock64();
	printf("Single Precision divide latency: %llu\n", end - start);
	
}

/*=============================================================
	       INTEGER INTRINSICS LATENCY KERNEL
=============================================================*/
__global__
void hadd_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 10000;
	int a = 10000;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = __hadd(a,c);
	end = clock64();
	printf("Integer Intrinsic average latency: %llu\n", end - start);
};

__global__
void mul24_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 1;
	int a = 1;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = __mul24(a,c);
	end = clock64();
	printf("Integer Intrisic multiply latency: %llu\n", end - start);
};

__global__
void sadd_latency()
{
	long long unsigned start;
	long long unsigned end;
	int c = 1;
	int a = 1;
	int b = 0'
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = __sad(a,c,b);
	end = clock64();
	printf("Interger Intrinsic absolute value latency: %llu\n", end - start);
};


/*=============================================================
	   SINGLE PRECISION INTRINSICS LATENCY KERNEL
=============================================================*/

__global__
void cosf_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = __cosf(c);
	end = clock64();
	printf("Single Precision Intrinsic cosine latency: %llu\n", end - start);
};

__global__
void sinf_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = __sinf(c);
	end = clock64();
	printf("Single Precision Intrinsic sine latency: %llu\n", end - start);
};

__global__
void fadd_rd_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	float a = 1.0
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = fadd_rd(a,c);
	end = clock64();
	printf("Single Precision Intrinsic add latency: %llu\n", end - start);
};

__global__
void fdiv_rd_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	float a = 1.0
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = fdiv_rd(a,c);
	end = clock64();
	printf("Single Precision Intrinsic divide latency: %llu\n", end - start);
};

__global__
void fmaf_rd_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	float a = 1.0
	float b = 0.0;
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = fmaf_rd(a,b,c);
	end = clock64();
	printf("Single Precision Intrinsic FMA latency: %llu\n", end - start);
};

__global__
void fmul_rd_latency()
{
	long long unsigned start;
	long long unsigned end;
	float c = 1.0;
	float a = 1.0
	start = clock64();
	#pragma unroll (STREAM_LENGTH)
	for (int i = 0 ; i < STREAM_LENGTH; i++)
		c = fmul_rd(a,c);
	end = clock64();
	printf("Single Precision Intrinsic multiply latency: %llu\n", end - start);
};



int main(void) 
{

	/*
		LATENCY
	*/

	// Integer Arithmetic
	i_add_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	i_min_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	i_mul_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	i_mod_latency<<<1,1>>>();
	cudaDeviceSynchronize();

	//Single precision Arithmetic
	f_add_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	f_min_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	f_mul_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	f_div_latency<<<1,1>>>();
	cudaDeviceSynchronize();

	// Integer Intrinsics
	hadd__latency<<<1,1>>>();
	cudaDeviceSynchronize();
	mul24_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	sad_latency<<<1,1>>>();
	cudaDeviceSynchronize();

	//Single precision Intrinsics;
	cosf_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	sinf_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	fadd_rd_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	fdiv_rd_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	fmaf_rd_latency<<<1,1>>>();
	cudaDeviceSynchronize();
	fmul_rd_latency<<<1,1>>>();

}
