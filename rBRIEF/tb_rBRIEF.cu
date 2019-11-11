#include<stdlib.h>
#include<iostream>
#include<chrono>
#include"rBRIEF.cuh"

/*=============*/
#define PRINTSTATS
/*=============*/

int main(int argc, char const *argv[]) {

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

  // 3) Run gpu kernel
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
  #endif
}
