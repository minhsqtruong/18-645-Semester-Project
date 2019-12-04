#include<stdlib.h>
#include<iostream>
#include<chrono>
#include"match.cuh"
#include <cuda_runtime.h>

/*=============*/
#define PRINTSTATS
/*=============*/

int main(int argc, char const *argv[]) {

  //CPU=========================================================================

  // 1) Initialized arguments
  int l1 = 60000;
  int l2 = 60000;
  int* binVectorArray1;
  int* binVectorArray2;
  int* result;
  cudaMallocManaged(&binVectorArray1, sizeof(int) * l1);
  cudaMallocManaged(&binVectorArray2, sizeof(int) * l2);
  cudaMallocManaged(&result, sizeof(int) * l1);
  for (int i = 0; i < l1; i++) {  
    binVectorArray1[i] = i;
    binVectorArray2[i] = i;
  }
  // 2) Run cpu reference
  auto t1 = std::chrono::high_resolution_clock::now();
  //cpu_match(l1, l2, binVectorArray1, binVectorArray2, result);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  
  #ifdef PRINTSTATS
  //std::cout << "CPU reference: " << std::endl;
  for (int i = 0; i < l1; i++) {
	//std::cout << result[i] << " ";
  }
  //int throughputC = (l1 * 4 + l2 * 4 )/ duration; 
  //std::cout << "CPU implementation takes: " << duration << " microseconds" <<std::endl;
   //std::cout << "GPU implementation : " << throughputC << "bytes/ microseconds" <<std::endl;
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
  cudaMallocManaged(&result, sizeof(int) * l1);
  // 5) Run gpu
  auto t3 = std::chrono::high_resolution_clock::now();
  int * distances;
  cudaMallocManaged(&distances, sizeof(int) * l1 * l2);
  gpu_match(l1, l2, binVectorArray1, binVectorArray2, result, distances);
  auto t4 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
  cudaDeviceSynchronize();  
  //std::cout << "GPU reference: " << std::endl;
  for (int i = 0; i < l1; i++) {
	//std::cout << result[i] << " ";
  }
  int throughput = (l1 * 4 + l2 * 4)/ duration2;  
  std::cout << "GPU implementation takes: " << duration2 << " microseconds" <<std::endl;
  std::cout << "GPU implementation : " << throughput << "bytes/ microseconds" <<std::endl;
  
}