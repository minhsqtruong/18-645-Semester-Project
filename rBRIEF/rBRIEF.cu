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
   // // 1) Shared memory management
   // extern __shared__ float4 shared[];
   // int4* sharedPattern = (int4*) shared;
   // float4* sharedPatches0 = (float4*) &shared[256];
   // float4* sharedPatches1 = (float4*) &shared[N*blockDim.x*24 + 256];
   // float4* thisPatches;
   // float4* nextPatches;
   // float4* tmp;
   //
   // // 2) Load pattern into shared memory (static part of kernel)
   // int id = threadIdx.x;
   // int stride = blockDim.x;
   // for (int i = id; i < 256; i+= stride) {
   //   sharedPattern[i] = pattern[i];
   // }
   //
   // // 3) Preload patches 0 into shared memory
   // int start = blockIdx.x * (N*24) + id;
   // int end   = blockIdx.x * (N*24) + N*24;
   // for (int i = start; i < end; i+=stride) {
   //   sharedPatches0[i] = patches[i];
   // }
   // thisPatches = sharedPatches0;

   // Kernel Loop begin:
   //for (int i = blockIdx.x; i < (P - 1) * N * blockDim.x*24; i+= )

 };

 /*============================================================================*/
 /*
conflict_free_index return the bank conflict free index
 */

 __device__ __forceinline__ int conflict_free_index(int local_id, int real_idx)
 {
   return real_idx * 128 + local_id;
 }

 /*============================================================================*/
 /*
 gpu_rBRIEF_naive naive implementation of kernel, serve as baseline upon which
 better kernel are design
 */
 __global__ void gpu_rBRIEF_naive(float4* patches, int4* pattern, int4* train_bin_vec, int K, int P)
 {
   // 0) Memory Setup
   extern __shared__ float shared_patchBank[];
   int4   private_pattern[32];
   // int4   train_bin_vec_buff0[2];
   // int4   train_bin_vec_buff1[2];
   // int4*  thisBuff, nextBuff, tmp;
   // unsigned int train_vec_x, train_vec_y, train_vec_z, train_vec_w;

   // coordinate initialize in Private Registers
   int coord[96] = { -0, -0, -0, -0, -0, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                     -0, -4, -3, -2, -1, 0};

   // 1) Setup thread ids
   int local_id = threadIdx.x;
   int global_id = blockIdx.x * gridDim.x + local_id;

   // 2) Load Sampling Pattern into Private Registers
   #pragma unroll
   for (int i = 0; i < 32; i++)
      private_pattern[i] = pattern[i];

   // 3) Load my patch into dedicated bank
   for (int itr = 0; itr < 1000; itr++) {
     float4 thisNum;
     #pragma unroll
     for (int i = 1; i < 24; i++) {
       thisNum = patches[global_id + (i / 4) * P];
       shared_patchBank[conflict_free_index(local_id, i)]       = thisNum.x;
       shared_patchBank[conflict_free_index(local_id, i*4 + 1)] = thisNum.y;
       shared_patchBank[conflict_free_index(local_id, i*4 + 2)] = thisNum.z;
       shared_patchBank[conflict_free_index(local_id, i*4 + 3)] = thisNum.w;
     }

     // 4) 1 thread works on 1 patch at a time
     float m01 = 0.0;
     float m10 = 0.0;
     float intensity;
     float theta;
     #pragma unroll
     for (int i = 5; i < 96; i++) {
      intensity = shared_patchBank[conflict_free_index(local_id, i)];
      m01       = __fmaf_rd(coord[i / 10], intensity, m01);
      m10       = __fmaf_rd(coord[i], intensity, m10);
     }
     theta = atan2f(m01, m10);
     #ifdef rBRIEFDEBUG
     if (threadIdx.x == 0)
      printf("m01: %f m10: %f theta: %f\n", m01, m10, theta);
     #endif

     // 5) Calculate the sin and cos of theta
     float sin, cos;
     sincosf(theta, &sin, &cos); // BOTTLE NECK!!!
     #ifdef rBRIEFDEBUG
     if (threadIdx.x == 0)
       printf("sin: %f cos: %f\n",sin, cos);
     #endif

     // 6) Sample the patch and return its binary vector
     float Ia, Ib;
     int ax, ay, bx, by;
     unsigned int idxa, idxb;
     int rotated_ax, rotated_ay, rotated_bx, rotated_by;
     unsigned int binVector = 0;
     int result;
     #pragma unroll
     for (int i = 0; i < 32; ++i) {
       ax = private_pattern[i].x;
       ay = private_pattern[i].y;
       bx = private_pattern[i].z;
       by = private_pattern[i].w;

       rotated_ax = (int) (cos * ax - sin * ay);
       rotated_ay = (int) (-10 * (sin * ay + cos * ay));
       rotated_bx = (int) (cos * bx - sin * by);
       rotated_by = (int) (-10 * (sin * by + cos * by));

       idxa = __sad(rotated_ax, rotated_ay, 0) % 96;
       idxb = __sad(rotated_bx, rotated_by, 0) % 96;

       Ia = shared_patchBank[conflict_free_index(local_id, idxa)];
       Ib = shared_patchBank[conflict_free_index(local_id, idxb)];

       result = ((int) Ia > Ib) << i;
       binVector |= result;
     }
     #ifdef rBRIEFDEBUG
     if (threadIdx.x == 0) {
       printf("My Binary vector is: ");
       while (binVector) {
          if (binVector & 1)
              printf("1");
          else
              printf("0");

          binVector >>= 1;
      }
      printf("\n");
    }
    #endif

     // 7) Preload binary vector from Global Memory and perform Hamming distance calculation
     // train_bin_vec_buff0[0] = train_bin_vec[0];
     // train_bin_vec_buff0[1] = train_bin_vec[1];
     // nextBuff = train_bin_vec_buff0;
     // for (int i = 1; i < 16; i++) {
     //   tmp = thisBuff;
     //   thisBuff = nextBuff;
     //   nextBuff = tmp;
     //   // Preload
     //   nextBuff[0] = train_bin_vec[i * 2 + 0];
     //   nextBuff[1] = train_bin_vec[i * 2 + 1];
     //
     //   // Calculate distance
     //   train_vec_x = thisBuff[0].x;
     //   train_vec_y = thisBuff[0].y;
     //   train_vec_z = thisBuff[0].z;
     //   train_vec_w = thisBuff[0].w;
     //
     //   train_vec_x ^= binVector;
     //   train_vec_y ^= binVector;
     //   train_vec_z ^= binVector;
     //   train_vec_w ^= binVector;
     //
     //   train_vec_x = __popc(train_vec_x);
     //   train_vec_y = __popc(train_vec_y);
     //   train_vec_z = __popc(train_vec_z);
     //   train_vec_w = __popc(train_vec_w);
     //
     //   // Store Back
     //
     //   train_vec_x = thisBuff[1].x;
     //   train_vec_y = thisBuff[1].y;
     //   train_vec_z = thisBuff[1].z;
     //   train_vec_w = thisBuff[1].w;
     //
     //   train_vec_x ^= binVector;
     //   train_vec_y ^= binVector;
     //   train_vec_z ^= binVector;
     //   train_vec_w ^= binVector;
     //
     //   train_vec_x = __popc(train_vec_x);
     //   train_vec_y = __popc(train_vec_y);
     //   train_vec_z = __popc(train_vec_z);
     //   train_vec_w = __popc(train_vec_w);

       // Store Back
     //}
  }
 }
 /*============================================================================*/
 /*
 gpu_oBRIEF
     @patches: global memory patches stored in float4 format
     @pattern: global memory patterns stored in float4 format
 */
 void gpu_rBRIEF(float4* patches, int4* pattern, int4* train_bin_vec, int K, int P)
 {
   int numBlocks =  1;
   int numThreads = 128;
   int sharedMemSize = 96 * 128 * sizeof(float);
   //gpu_rBRIEF_Loop<<<numBlocks, numThreads,shared_size>>>(N, patches, pattern);
   gpu_rBRIEF_naive<<<numBlocks, numThreads, sharedMemSize>>>(patches, pattern, train_bin_vec, K, P);
   cudaDeviceSynchronize();
 };

/*============================================================================*/
/*
pipeline_print_rBRIEF is just a testing function
*/
void pipeline_print_rBRIEF(){ printf("rBRIEF Module active!\n");};

