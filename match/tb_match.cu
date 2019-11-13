#include<stdlib.h>
#include<iostream>
#include<chrono>
#include"match.cuh"

/*=============*/
#define PRINTSTATS
/*=============*/

int main(int argc, char const *argv[]) {

  //CPU=========================================================================

  // 1) Initialized arguments
  int l1 = 10;
  int l2 = 10;
  int width = 256; 
  bool* binVectorArray1 = (bool*) malloc(sizeof(bool) * l1 * width);
  bool* binVectorArray2 = (bool*) malloc(sizeof(bool) * l2 * width);
  int* result = (int*) malloc(sizeof(int) * l1);
  for (int i = 0; i < l1 * width; i++) {
    bool val = (rand() % 10) > 5;  
    binVectorArray1[i] = val;
    binVectorArray2[i] = val;
  }
  // 2) Run cpu reference
  auto t1 = std::chrono::high_resolution_clock::now();
  cpu_match(width, l1, l1, binVectorArray1, binVectorArray2, result);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  
  #ifdef PRINTSTATS
  std::cout << "CPU reference: " << std::endl;
  for (int i = 0; i < l1; i++) {
	std::cout << result[i] << " ";
  }
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
  bool* binVectorArray3;
  bool* binVectorArray4;
  cudaMallocManaged(&binVectorArray3, sizeof(bool) * l1 * width);
  cudaMallocManaged(&binVectorArray4, sizeof(bool) * l2 * width);
  for (int i = 0; i < l1 * width; i++) {
    binVectorArray3[i] = (rand() % 1) > 0; 
  }
  for (int i = 0; i < l2 * width; i++) {
    binVectorArray4[i] = (rand() % 1) > 0; 
  }
  // 5) Run gpu
  gpu_match(width,l1, l2, binVectorArray3, binVectorArray4);
}