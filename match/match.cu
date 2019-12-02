#include "stdio.h"
#include "cuda_runtime_api.h"

//------------------------------------------------------------------------------
// DEVICE FUNCTIONS



// write implementation of match module here

/*============================================================================*/
/*
    @binVectorArray1: size l1 x width 
	@binVectorArray2: size l2 x width 
	@result: size l1 
*/
void cpu_match(int width, int l1, int l2, bool* binVectorArray1, bool* binVectorArray2, int * result)
{
  for (int i = 0; i < l1; i +=1) {
	int bestJ = 0;
	int bestDist = width + 1; 
	bool * binVector1 = &binVectorArray1[i*width];
	for (int j = 0; j < l2; j += 1) {
		bool * binVector2 = &binVectorArray2[j*width];
		int curDist = 0; 
		for (int k = 0; k < width; k += 1){
			curDist += (int) (binVector1[k] != binVector2[k]);  
		}
		if (curDist < bestDist ) {
			bestDist = curDist;
			bestJ = j;
		}
	}
	
	result[i] = bestJ; 
  }

};
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*============================================================================*/
/*
gpu_match_Loop iteratively execute the kernel until done
    @binVectorArray1: size l1 x width 
*/
 __global__ void gpu_match_Loop(int width, int l1, int l2, bool* binVectorArray1,bool* binVectorArray2, int* result, int* distances)
 {
   //printf("index: %d\n", width);
   //__shared__ int distances[10000];
   //extern __shared__ bool BV2[];
   //extern __shared__ bool BV1[];
   int y, x, i, j;
   // Determine thread row y and column x within thread block.
   y = threadIdx.y;
   x = threadIdx.x;
   // Determine matrix element row i and column j.
   i = blockIdx.y*blockDim.y + y;
   j = blockIdx.x*blockDim.x + x;
   //printf("i: %d\n", i);

   
   if (i >= l1 || j >= l2) {
      return; 
   }
     
   // Each thread computes its own matrix element.
   bool * binVector1 = &binVectorArray1[i*width];
   bool * binVector2 = &binVectorArray2[j*width]; 
  
   int curDist = 0; 
   for (int k = 0; k < width; k += 1){
 	curDist += (int) (binVector1[k] != binVector2[k]);  	
   }
  
   distances[i * l2 + j] = curDist;
   __syncthreads();
   if (i == 0 && j ==0 ) {
      for (int p = 0; p < l1; p++) {
	//printf("\n");
	for (int q = 0; q < l2; q++) {
	    if (distances[p * l2 + q] == 0) {
	       //printf("%d  %d \n", p, q );
	    }
  	}
       }
   }
   if (j == 0) {
      	int bestJ = 0;
	int bestDist = width + 1; 
   	for (int k = 0; k < l2; k += 1) {
	    	if (distances[i * l2 + k]  < bestDist ) {
		   	if (i != k && (k == 20 || k == 10 )) {
		   	printf("check7777: %d %d %d %d\n", distances[i * l2 + k],i, k, bestDist);
}
			bestDist = distances[i * l2 + k];
			bestJ = k;
		}
	}
	result[i] = bestJ; 	 
   }

   

 };


 /*============================================================================*/
 /*
    @binVectorArray1: size l1 x width 
	@binVectorArray2: size l2 x width  
 */
 void gpu_match(int width, int l1, int l2, bool* binVectorArray1, bool* binVectorArray2, int* result)
 { 
   int * distances;
   cudaMallocManaged(&distances, sizeof(int) * l1 * l2);
   dim3 GRID_DIM (l1/16, l2/16);
   dim3 BLOCK_DIM (16, 16);
   gpu_match_Loop<<<GRID_DIM, BLOCK_DIM>>>(width, l1, l2, binVectorArray1, binVectorArray2, result, distances);
   gpuErrchk( cudaPeekAtLastError() );
   gpuErrchk( cudaDeviceSynchronize() );
 };

/*============================================================================*/
/*
pipeline_print_match is just a testing function
*/
void pipeline_print_match(){ printf("Match Module active!\n");};

