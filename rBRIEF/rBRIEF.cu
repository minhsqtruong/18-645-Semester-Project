#include "rBRIEF.cuh"
/*
Specification:
    The Kernel for oBRIEF is designed for the Quadro P2000 GPU with:

    CPU-GPU interface:
      Global Memory Bandwidth: 140 GB/s
      GPU clock: 1.37 GHz
      Global Memory Size: 5.12 GB
      Memory Bus: 160 bits / 20 bytes

    Size:
      SM Count: 8
      CUDA cores: 1024 (4 warps per SM)
      Threads: 32 per Warp
      Registers: 65536 per SM
      Shared Memory: 49152 per SM
      L1 Cache: 48 KB per SM
      L2 Cache: 1280 KB

    Memory Hierachy:
      1) Register File
      2) Shared Memory - L1 cache
      3) L2 cache
      4) Global Memory
*/

/*============================================================================*/
/*
cpu_oBRIEF calculate binary descriptor for the patches.
    @numPatch: number of patches that needs binary vector calculation
    @patchDim: the side dimension of each patch
    @patchArray:  a 1D array that holds the patches in consecutive order.
    @binVectorArray: a 1D boolean vector that holds all binary descriptors
    @pattern: the precomputed pattern use for binary sampling.
*/
void cpu_rBRIEF(int numPatch, int patchDim, float* patchArray, bool* binVectorArray, int* pattern)
{
  for (int patchIdx = 0; patchIdx < numPatch; patchIdx+=1) {

    // 1) Get the patch and its binary vector
    float * patch = &patchArray[patchIdx*(patchDim*patchDim)];
    int patchCenter = patchDim / 2;
    bool * binVector = &binVectorArray[patchIdx*256];

    // 2) Calculate the angle for the patch
    float m01 = 0.f;
    float m10 = 0.f;
    float theta;
    for (int pixIdx = 0; pixIdx < patchDim*patchDim; pixIdx++) {
      int x = pixIdx % patchDim*patchDim;
      int y = pixIdx / patchDim*patchDim;
      m01 += (y - patchCenter) * patch[pixIdx]; // offset so that center is origin
      m10 += (x - patchCenter) * patch[pixIdx]; // offset so that center is origin
    }
    theta = atan2f(m01, m10);

    // 3) Calculate sin, cos of the angle
    float sinTheta, cosTheta;
    sincosf(theta, &sinTheta, &cosTheta);

    // 4) Sample the patch and return its binary feature
    float Ia, Ib;
    int ax, ay, bx, by;
    int rotated_ax, rotated_ay, rotated_bx, rotated_by;
    for (int i = 0; i < 256; ++i) {
      ax = pattern[4*i];
      ay = pattern[4*i+1];
      bx = pattern[4*i+2];
      by = pattern[4*i+3];

      rotated_ax = (int) (cosTheta * ax - sinTheta * ay);
      rotated_ay = (int) (sinTheta * ay + cosTheta * ay);
      rotated_bx = (int) (cosTheta * bx - sinTheta * by);
      rotated_by = (int) (sinTheta * by + cosTheta * by);

      Ia = patch[rotated_ax + patchDim * rotated_ay];
      Ib = patch[rotated_bx + patchDim * rotated_by];

      binVector[i] = Ia > Ib;
    }
  }
};

/*============================================================================*/
/*
gpu_oBRIEF_Kernel
*/
// __device__ void gpu_oBRIEF_Kernel()
// {
//
// };

/*============================================================================*/
/*
gpu_oBRIEF_Loop iteratively execute the kernel until done
    @N: number of patches per thread used to compute N angle.
    @patches: global memory patches stored in float4 format
    @pattern: global memory patterns stored in float4 format
*/
 __global__ void gpu_rBRIEF_Loop(int N, float4* patches, int4* pattern)
 {
   // 1) Shared memory management
   extern __shared__ float4 shared[];
   int4* sharedPattern = (int4*) shared;
   float4* sharedPatches0 = (float4*) &shared[256];
   float4* sharedPatches1 = (float4*) &shared[N*blockDim.x*24 + 256];
   float4* thisPatches;
   float4* nextPatches;
   float4* tmp;

   // 2) Load pattern into shared memory (static part of kernel)
   int id = threadIdx.x;
   int stride = blockDim.x;
   for (int i = id; i < 256; i+= stride) {
     sharedPattern[i] = pattern[i];
   }

   // 3) Preload patches 0 into shared memory
   int start = blockIdx.x * (N*24) + id;
   int end   = blockIdx.x * (N*24) + N*24;
   for (int i = start; i < end; i+=stride) {
     sharedPatches0[i] = patches[i];
   }
   thisPatches = sharedPatches0;

   // Kernel Loop begin:
   //for (int i = blockIdx.x; i < (P - 1) * N * blockDim.x*24; i+= )

 };


 /*============================================================================*/
 /*
 gpu_oBRIEF
     @patches: global memory patches stored in float4 format
     @pattern: global memory patterns stored in float4 format
 */
 void gpu_rBRIEF(float4* patches, int4* pattern)
 {
   int N = 1;
   int numBlocks =  10;
   int numThreads = 128;
   int shared_size = sizeof(float4) * (256 + N*numThreads*2);
   gpu_rBRIEF_Loop<<<numBlocks, numThreads,shared_size>>>(N, patches, pattern);
 };

/*============================================================================*/
/*
pipeline_print_rBRIEF is just a testing function
*/
void pipeline_print_rBRIEF(){ printf("rBRIEF Module active!\n");};
