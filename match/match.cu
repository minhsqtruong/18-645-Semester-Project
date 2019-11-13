#include "stdio.h"

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


/*============================================================================*/
/*
gpu_match_Loop iteratively execute the kernel until done
    @N: number of keypoints per thread to compute best match for.
    @binVectorArray1: size l1 x width 
	@binVectorArray2: size l2 x width  
*/
 __global__ void gpu_match_Loop(int N, int width, int l1, int l2, bool* binVectorArray1, bool* binVectorArray2)
 {

   extern __shared__ int shared[];
   int id = threadIdx.x;
   int start = id * N; 
   int max = id * N + N; 
   if (max > l1) {
	max = l1; 
   } 
   if (start > l1) {
	start = l1; 
   }
   for (int i = id * N; i < max; i +=1) {
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
   shared[i] = bestJ; 
   }


 };


 /*============================================================================*/
 /*
    @binVectorArray1: size l1 x width 
	@binVectorArray2: size l2 x width  
 */
 void gpu_match(int width, int l1, int l2, bool* binVectorArray1, bool* binVectorArray2)
 {
   int N = 1;
   int numBlocks =  10;
   int numThreads = 128;
   if (numThreads > l1) {
	numThreads = l1;
   }
   N = numThreads / l1;
   int shared_size = sizeof(int) * (l1);
   gpu_match_Loop<<<numBlocks, numThreads,shared_size>>>(N, width, l1, l2, binVectorArray1, binVectorArray2);
 };

/*============================================================================*/
/*
pipeline_print_match is just a testing function
*/
void pipeline_print_match(){ printf("Match Module active!\n");};

