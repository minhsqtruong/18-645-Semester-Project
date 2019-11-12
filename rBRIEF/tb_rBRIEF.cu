#include<stdlib.h>
#include<iostream>
#include<chrono>
#include"rBRIEF.cuh"

/*=============*/
#define PRINTSTATS
/*=============*/

int main(int argc, char const *argv[]) {

  //CPU=========================================================================

  // 1) Initialized arguments
  int numPatch = 10;
  int patchDim = 10;
  float* patchArray = (float*) malloc(sizeof(float) * numPatch * patchDim * patchDim);
  bool* binVectorArray = (bool*) malloc(sizeof(bool) * numPatch * 256);
  for (int i = 0; i < numPatch * patchDim * patchDim; i++) {
    patchArray[i] = static_cast <float> (rand()) / static_cast <float> (255.0);
  }
  extern int cpu_precompute_BRIEF_pattern[256*4];
  int* pattern = cpu_precompute_BRIEF_pattern;

  // 2) Run cpu reference
  auto t1 = std::chrono::high_resolution_clock::now();
  cpu_oBRIEF(numPatch, patchDim, patchArray, binVectorArray, pattern);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

  #ifdef PRINTSTATS
  std::cout << "CPU reference: " << std::endl;
  printMatrix<bool*>(binVectorArray, numPatch, 256);
  std::cout << "CPU implementation takes: " << duration << " microseconds" <<std::endl;
  #endif

  //GPU=========================================================================

  // 3) Check GPU stats
  #ifdef PRINTSTATS
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << std::endl;
  std::cout << "GPU Name: " << prop.name << std::endl;
  std::cout << "Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
  std::cout << "Shared Memory per SM: " << prop.sharedMemPerBlock << " bytes" << std::endl;
  std::cout << "Registers per SM: " << prop.regsPerBlock << std::endl;
  std::cout << "Warp Size:  " << prop.warpSize << std::endl;
  std::cout << "Number of SM: " << prop.multiProcessorCount << std::endl;
  std::cout << std::endl;
  #endif

  // 4) GPU initialization, memory management
  int P = 10000;
  float4 * gpu_patches;
  int4* gpu_pattern;
  cudaMallocManaged(&gpu_patches, sizeof(float4) * 24 * P);
  cudaMallocManaged(&gpu_pattern, sizeof(float4) * 256);
  for (int i = 0; i < P * 24; i++) {
    float x = static_cast <float> (rand()) / static_cast <float> (255.0);
    float y = static_cast <float> (rand()) / static_cast <float> (255.0);
    float z = static_cast <float> (rand()) / static_cast <float> (255.0);
    float w = static_cast <float> (rand()) / static_cast <float> (255.0);
    gpu_patches[i] = make_float4(x,y,z,w);
  }
  for (int i = 0; i < 256; i++) {
    int x = cpu_precompute_BRIEF_pattern[i*4 + 0];
    int y = cpu_precompute_BRIEF_pattern[i*4 + 1];
    int z = cpu_precompute_BRIEF_pattern[i*4 + 2];
    int w = cpu_precompute_BRIEF_pattern[i*4 + 3];
    gpu_pattern[i] = make_int4(x,y,z,w);
  }

  // 5) Run gpu
  gpu_oBRIEF(gpu_patches, gpu_pattern);
}
