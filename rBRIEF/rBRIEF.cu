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
 calculate_moments calculate the moments of a patch
 */
 __forceinline __device__ void calculate_moments(float4* patches,
                                                 float* _m01,
                                                 float* _m10,
                                                 int P,
                                                 int global_id,
                                                 float4 intensity0,
                                                 float4 intensity1,
                                                 float4 intensity2,
                                                 float4 intensity3,
                                                 float4 intensity4,
                                                 float4 intensity5,
                                                 float4 intensity6,
                                                 float4 intensity7,
                                                 float4 intensity8,
                                                 float4 intensity9,
                                                 float4 intensity10,
                                                 float4 intensity11,
                                                 float4 intensity12,
                                                 float4 intensity13,
                                                 float4 intensity14,
                                                 float4 intensity15,
                                                 float4 intensity16,
                                                 float4 intensity17,
                                                 float4 intensity18,
                                                 float4 intensity19,
                                                 float4 intensity20,
                                                 float4 intensity21,
                                                 float4 intensity22,
                                                 float4 intensity23,
                                                 )
 {
   float m01 = 0.0;
   float m10 = 0.0;
   float4 intensity;

   intensity = intensity0 ;
   m01 = __fmaf_rd(-5.0, intensity.x,m01);
   m10 = __fmaf_rd(-5.0, intensity.x,m10);
   m01 = __fmaf_rd(-4.0, intensity.y,m01);
   m10 = __fmaf_rd(-5.0, intensity.y,m10);
   m01 = __fmaf_rd(-3.0, intensity.z,m01);
   m10 = __fmaf_rd(-5.0, intensity.z,m10);
   m01 = __fmaf_rd(-2.0, intensity.w,m01);
   m10 = __fmaf_rd(-5.0, intensity.w,m10);
   intensity = intensity1 ;
   m01 = __fmaf_rd(-1.0, intensity.x,m01);
   m10 = __fmaf_rd(-5.0, intensity.x,m10);
   //m01 = __fmaf_rd(-0.0, intensity.y,m01);
   m10 = __fmaf_rd(-5.0, intensity.y,m10);
   m01 = __fmaf_rd( 1.0, intensity.z,m01);
   m10 = __fmaf_rd(-5.0, intensity.z,m10);
   m01 = __fmaf_rd( 2.0, intensity.w,m01);
   m10 = __fmaf_rd(-5.0, intensity.w,m10);
   intensity = intensity2 ;
   m01 = __fmaf_rd( 3.0, intensity.x,m01);
   m10 = __fmaf_rd(-5.0, intensity.x,m10);
   m01 = __fmaf_rd( 4.0, intensity.y,m01);
   m10 = __fmaf_rd(-5.0, intensity.y,m10);

   m01 = __fmaf_rd(-5.0, intensity.z,m01);
   m10 = __fmaf_rd(-4.0, intensity.z,m10);
   m01 = __fmaf_rd(-4.0, intensity.w,m01);
   m10 = __fmaf_rd(-4.0, intensity.w,m10);
   intensity = intensity3 ;
   m01 = __fmaf_rd(-3.0, intensity.x,m01);
   m10 = __fmaf_rd(-4.0, intensity.x,m10);
   m01 = __fmaf_rd(-2.0, intensity.y,m01);
   m10 = __fmaf_rd(-4.0, intensity.y,m10);
   m01 = __fmaf_rd(-1.0, intensity.z,m01);
   m10 = __fmaf_rd(-4.0, intensity.z,m10);
   //m01 = __fmaf_rd(-0.0, intensity.w,m01);
   m10 = __fmaf_rd(-4.0, intensity.w,m10);
   intensity = intensity4 ;
   m01 = __fmaf_rd( 1.0, intensity.x,m01);
   m10 = __fmaf_rd(-4.0, intensity.x,m10);
   m01 = __fmaf_rd( 2.0, intensity.y,m01);
   m10 = __fmaf_rd(-4.0, intensity.y,m10);
   m01 = __fmaf_rd( 3.0, intensity.z,m01);
   m10 = __fmaf_rd(-4.0, intensity.z,m10);
   m01 = __fmaf_rd( 4.0, intensity.w,m01);
   m10 = __fmaf_rd(-4.0, intensity.w,m10);

   intensity = intensity5 ;
   m01 = __fmaf_rd(-5.0, intensity.x,m01);
   m10 = __fmaf_rd(-3.0, intensity.x,m10);
   m01 = __fmaf_rd(-4.0, intensity.y,m01);
   m10 = __fmaf_rd(-3.0, intensity.y,m10);
   m01 = __fmaf_rd(-3.0, intensity.z,m01);
   m10 = __fmaf_rd(-3.0, intensity.z,m10);
   m01 = __fmaf_rd(-2.0, intensity.w,m01);
   m10 = __fmaf_rd(-3.0, intensity.w,m10);
   intensity = intensity6 ;
   m01 = __fmaf_rd(-1.0, intensity.x,m01);
   m10 = __fmaf_rd(-3.0, intensity.x,m10);
   //m01 = __fmaf_rd(-0.0, intensity.y,m01);
   m10 = __fmaf_rd(-3.0, intensity.y,m10);
   m01 = __fmaf_rd( 1.0, intensity.z,m01);
   m10 = __fmaf_rd(-3.0, intensity.z,m10);
   m01 = __fmaf_rd( 2.0, intensity.w,m01);
   m10 = __fmaf_rd(-3.0, intensity.w,m10);
   intensity = intensity7 ;
   m01 = __fmaf_rd( 3.0, intensity.x,m01);
   m10 = __fmaf_rd(-3.0, intensity.x,m10);
   m01 = __fmaf_rd( 4.0, intensity.y,m01);
   m10 = __fmaf_rd(-3.0, intensity.y,m10);

   m01 = __fmaf_rd(-5.0, intensity.z,m01);
   m10 = __fmaf_rd(-2.0, intensity.z,m10);
   m01 = __fmaf_rd(-4.0, intensity.w,m01);
   m10 = __fmaf_rd(-2.0, intensity.w,m10);
   intensity = intensity8 ;
   m01 = __fmaf_rd(-3.0, intensity.x,m01);
   m10 = __fmaf_rd(-2.0, intensity.x,m10);
   m01 = __fmaf_rd(-2.0, intensity.y,m01);
   m10 = __fmaf_rd(-2.0, intensity.y,m10);
   m01 = __fmaf_rd(-1.0, intensity.z,m01);
   m10 = __fmaf_rd(-2.0, intensity.z,m10);
   //m01 = __fmaf_rd(-0.0, intensity.w,m01);
   m10 = __fmaf_rd(-2.0, intensity.w,m10);
   intensity = intensity9 ;
   m01 = __fmaf_rd( 1.0, intensity.x,m01);
   m10 = __fmaf_rd(-2.0, intensity.x,m10);
   m01 = __fmaf_rd( 2.0, intensity.y,m01);
   m10 = __fmaf_rd(-2.0, intensity.y,m10);
   m01 = __fmaf_rd( 3.0, intensity.z,m01);
   m10 = __fmaf_rd(-2.0, intensity.z,m10);
   m01 = __fmaf_rd( 4.0, intensity.w,m01);
   m10 = __fmaf_rd(-2.0, intensity.w,m10);

   intensity = intensity10 ;
   m01 = __fmaf_rd(-5.0, intensity.x,m01);
   m10 = __fmaf_rd(-1.0, intensity.x,m10);
   m01 = __fmaf_rd(-4.0, intensity.y,m01);
   m10 = __fmaf_rd(-1.0, intensity.y,m10);
   m01 = __fmaf_rd(-3.0, intensity.z,m01);
   m10 = __fmaf_rd(-1.0, intensity.z,m10);
   m01 = __fmaf_rd(-2.0, intensity.w,m01);
   m10 = __fmaf_rd(-1.0, intensity.w,m10);
   intensity = intensity11 ;
   m01 = __fmaf_rd(-1.0, intensity.x,m01);
   m10 = __fmaf_rd(-1.0, intensity.x,m10);
   //m01 = __fmaf_rd(-0.0, intensity.y,m01);
   m10 = __fmaf_rd(-1.0, intensity.y,m10);
   m01 = __fmaf_rd( 1.0, intensity.z,m01);
   m10 = __fmaf_rd(-1.0, intensity.z,m10);
   m01 = __fmaf_rd( 2.0, intensity.w,m01);
   m10 = __fmaf_rd(-1.0, intensity.w,m10);
   intensity = intensity12 ;
   m01 = __fmaf_rd( 3.0, intensity.x,m01);
   m10 = __fmaf_rd(-1.0, intensity.x,m10);
   m01 = __fmaf_rd( 4.0, intensity.y,m01);
   m10 = __fmaf_rd(-1.0, intensity.y,m10);

   m01 = __fmaf_rd(-5.0, intensity.z,m01);
   //m10 = __fmaf_rd( 0.0, intensity.z,m10);
   m01 = __fmaf_rd(-4.0, intensity.w,m01);
   //m10 = __fmaf_rd( 0.0, intensity.w,m10);
   intensity = intensity13 ;
   m01 = __fmaf_rd(-3.0, intensity.x,m01);
   //m10 = __fmaf_rd( 0.0, intensity.x,m10);
   m01 = __fmaf_rd(-2.0, intensity.y,m01);
   //m10 = __fmaf_rd( 0.0, intensity.y,m10);
   m01 = __fmaf_rd(-1.0, intensity.z,m01);
   //m10 = __fmaf_rd( 0.0, intensity.z,m10);
   //m01 = __fmaf_rd(-0.0, intensity.w,m01);
   //m10 = __fmaf_rd( 0.0, intensity.w,m10);
   intensity = intensity14 ;
   m01 = __fmaf_rd( 1.0, intensity.x,m01);
   //m10 = __fmaf_rd( 0.0, intensity.x,m10);
   m01 = __fmaf_rd( 2.0, intensity.y,m01);
   //m10 = __fmaf_rd( 0.0, intensity.y,m10);
   m01 = __fmaf_rd( 3.0, intensity.z,m01);
   //m10 = __fmaf_rd( 0.0, intensity.z,m10);
   m01 = __fmaf_rd( 4.0, intensity.w,m01);
   //m10 = __fmaf_rd( 0.0, intensity.w,m10);

   intensity = intensity15 ;
   m01 = __fmaf_rd(-5.0, intensity.x,m01);
   m10 = __fmaf_rd( 1.0, intensity.x,m10);
   m01 = __fmaf_rd(-4.0, intensity.y,m01);
   m10 = __fmaf_rd( 1.0, intensity.y,m10);
   m01 = __fmaf_rd(-3.0, intensity.z,m01);
   m10 = __fmaf_rd( 1.0, intensity.z,m10);
   m01 = __fmaf_rd(-2.0, intensity.w,m01);
   m10 = __fmaf_rd( 1.0, intensity.w,m10);
   intensity = intensity16 ;
   m01 = __fmaf_rd(-1.0, intensity.x,m01);
   m10 = __fmaf_rd( 1.0, intensity.x,m10);
   //m01 = __fmaf_rd(-0.0, intensity.y,m01);
   m10 = __fmaf_rd( 1.0, intensity.y,m10);
   m01 = __fmaf_rd( 1.0, intensity.z,m01);
   m10 = __fmaf_rd( 1.0, intensity.z,m10);
   m01 = __fmaf_rd( 2.0, intensity.w,m01);
   m10 = __fmaf_rd( 1.0, intensity.w,m10);
   intensity = intensity17 ;
   m01 = __fmaf_rd( 3.0, intensity.x,m01);
   m10 = __fmaf_rd( 1.0, intensity.x,m10);
   m01 = __fmaf_rd( 4.0, intensity.y,m01);
   m10 = __fmaf_rd( 1.0, intensity.y,m10);

   m01 = __fmaf_rd(-5.0, intensity.z,m01);
   m10 = __fmaf_rd( 2.0, intensity.z,m10);
   m01 = __fmaf_rd(-4.0, intensity.w,m01);
   m10 = __fmaf_rd( 2.0, intensity.w,m10);
   intensity = intensity18 ;
   m01 = __fmaf_rd(-3.0, intensity.x,m01);
   m10 = __fmaf_rd( 2.0, intensity.x,m10);
   m01 = __fmaf_rd(-2.0, intensity.y,m01);
   m10 = __fmaf_rd( 2.0, intensity.y,m10);
   m01 = __fmaf_rd(-1.0, intensity.z,m01);
   m10 = __fmaf_rd( 2.0, intensity.z,m10);
   //m01 = __fmaf_rd(-0.0, intensity.w,m01);
   m10 = __fmaf_rd( 2.0, intensity.w,m10);
   intensity = intensity19 ;
   m01 = __fmaf_rd( 1.0, intensity.x,m01);
   m10 = __fmaf_rd( 2.0, intensity.x,m10);
   m01 = __fmaf_rd( 2.0, intensity.y,m01);
   m10 = __fmaf_rd( 2.0, intensity.y,m10);
   m01 = __fmaf_rd( 3.0, intensity.z,m01);
   m10 = __fmaf_rd( 2.0, intensity.z,m10);
   m01 = __fmaf_rd( 4.0, intensity.w,m01);
   m10 = __fmaf_rd( 2.0, intensity.w,m10);

   intensity = intensity20 ;
   m01 = __fmaf_rd(-5.0, intensity.x,m01);
   m10 = __fmaf_rd(-3.0, intensity.x,m10);
   m01 = __fmaf_rd(-4.0, intensity.y,m01);
   m10 = __fmaf_rd(-3.0, intensity.y,m10);
   m01 = __fmaf_rd(-3.0, intensity.z,m01);
   m10 = __fmaf_rd(-3.0, intensity.z,m10);
   m01 = __fmaf_rd(-2.0, intensity.w,m01);
   m10 = __fmaf_rd(-3.0, intensity.w,m10);
   intensity = intensity21 ;
   m01 = __fmaf_rd(-1.0, intensity.x,m01);
   m10 = __fmaf_rd(-3.0, intensity.x,m10);
   //m01 = __fmaf_rd(-0.0, intensity.y,m01);
   m10 = __fmaf_rd(-3.0, intensity.y,m10);
   m01 = __fmaf_rd( 1.0, intensity.z,m01);
   m10 = __fmaf_rd( 3.0, intensity.z,m10);
   m01 = __fmaf_rd( 2.0, intensity.w,m01);
   m10 = __fmaf_rd( 3.0, intensity.w,m10);
   intensity = intensity22 ;
   m01 = __fmaf_rd( 3.0, intensity.x,m01);
   m10 = __fmaf_rd( 3.0, intensity.x,m10);
   m01 = __fmaf_rd( 4.0, intensity.y,m01);
   m10 = __fmaf_rd( 3.0, intensity.y,m10);

   m01 = __fmaf_rd(-5.0, intensity.z,m01);
   m10 = __fmaf_rd( 4.0, intensity.z,m10);
   m01 = __fmaf_rd(-4.0, intensity.w,m01);
   m10 = __fmaf_rd( 4.0, intensity.w,m10);
   intensity = intensity23 ;
   m01 = __fmaf_rd(-3.0, intensity.x,m01);
   m10 = __fmaf_rd( 4.0, intensity.x,m10);
   m01 = __fmaf_rd(-2.0, intensity.y,m01);
   m10 = __fmaf_rd( 4.0, intensity.y,m10);
   m01 = __fmaf_rd(-1.0, intensity.z,m01);
   m10 = __fmaf_rd( 4.0, intensity.z,m10);
   //m01 = __fmaf_rd(-0.0, intensity.w,m01);
   m10 = __fmaf_rd( 4.0, intensity.w,m10);
   m01 = __fmaf_rd( 1.0, intensity.x,m01);
   m10 = __fmaf_rd( 4.0, intensity.x,m10);
   m01 = __fmaf_rd( 2.0, intensity.y,m01);
   m10 = __fmaf_rd( 4.0, intensity.y,m10);
   m01 = __fmaf_rd( 3.0, intensity.z,m01);
   m10 = __fmaf_rd( 4.0, intensity.z,m10);
   m01 = __fmaf_rd( 4.0, intensity.w,m01);
   m10 = __fmaf_rd( 4.0, intensity.w,m10);

    *_m01 = m01;
    *_m10 = m10;
 }

 /*============================================================================*/
 /*
 gpu_rBRIEF_naive naive implementation of kernel, serve as baseline upon which
 better kernel are design
 */
 __global__ void gpu_rBRIEF_naive(float4* patches, int4* pattern, double4* train_bin_vec, int K, int P)
 {
   extern __shared__ float4 shared_mem[];
   int4* shared_pattern = (int4*) shared_mem;
   double4* shared_train_vec = (double4*) sharedPattern[256];

   // 1) Setup thread ids and stride
   int local_id = threadIdx.x;
   int local_stride = blockDim.x;
   int global_id = blockIdx.x * gridDim.x + local_id;
   int global_stride = blockDim.x * gridDim.x;

   // 2) Load Sampling Pattern into Shared Memory
   for (int i = local_id; i < 256; i+=local_stride)
      shared_pattern[i] = pattern[i];

   // 3) Load Training binary vector into Shared Memory
   for (int i = local_id; i < K; i+=local_stride)
      shared_train_vec[i] = train_bin_vec[i];

   // 4) 1 thread works on 1 patch at a time.
   float m01;
   float m10;
   float theta;
   float4 intensity0,
   intensity1,
   intensity2,
   intensity3,
   intensity4,
   intensity5,
   intensity6,
   intensity7,
   intensity8,
   intensity9,
   intensity10,
   intensity11,
   intensity12,
   intensity13,
   intensity14,
   intensity15,
   intensity16,
   intensity17,
   intensity18,
   intensity19,
   intensity20,
   intensity21,
   intensity22,
   intensity23;


   calculate_moments(patches,
                     &m01,
                     &m10,
                     P,
                     global_id,
                    intensity0,
                    intensity1,
                    intensity2,
                    intensity3,
                    intensity4,
                    intensity5,
                    intensity6,
                    intensity7,
                    intensity8,
                    intensity9,
                    intensity10,
                    intensity11,
                    intensity12,
                    intensity13,
                    intensity14,
                    intensity15,
                    intensity16,
                    intensity17,
                    intensity18,
                    intensity19,
                    intensity20,
                    intensity21,
                    intensity22,
                    intensity23);
   theta = atan2f(m01, m10);

   // 5) Calculate the sin and cos of theta
   float sin, cos;
   sincosf(theta, &sin, &cos); // BOTTLE NECK!!!

   // 6) Sample the patch and return its binary vector
   // float Ia, Ib;
   // int ax, ay, bx, by;
   // int rotated_ax, rotated_ay, rotated_bx, rotated_by;
   // for (int i = 0; i < 256; ++i) {
   //   ax = shared_pattern[4*i].x;
   //   ay = shared_pattern[4*i].y;
   //   bx = shared_pattern[4*i].z;
   //   by = shared_pattern[4*i].w;
   //
   //   rotated_ax = (int) (cos * ax - sin * ay);
   //   rotated_ay = (int) (sin * ay + cos * ay);
   //   rotated_bx = (int) (cos * bx - sin * by);
   //   rotated_by = (int) (sin * by + cos * by);
   //
   //   Ia = patch[rotated_ax + patchDim * rotated_ay];
   //   Ib = patch[rotated_bx + patchDim * rotated_by];
   //
   //   binVector[i] = Ia > Ib;
   // }

 }
 /*============================================================================*/
 /*
 gpu_oBRIEF
     @patches: global memory patches stored in float4 format
     @pattern: global memory patterns stored in float4 format
 */
 void gpu_rBRIEF(float4* patches, int4* pattern, double4* train_bin_vec, int K, int P)
 {
   int numBlocks =  10;
   int numThreads = 128;
   int shared_size = sizeof(float4) * (256) + sizeof(double4) * K;
   //gpu_rBRIEF_Loop<<<numBlocks, numThreads,shared_size>>>(N, patches, pattern);
   gpu_rBRIEF_naive<<<numBlocks, numThreads, shared_size>>>(patches, pattern, train_bin_vec, K, P);
 };

/*============================================================================*/
/*
pipeline_print_rBRIEF is just a testing function
*/
void pipeline_print_rBRIEF(){ printf("rBRIEF Module active!\n");};
