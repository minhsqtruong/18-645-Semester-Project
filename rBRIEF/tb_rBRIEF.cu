#include<stdlib.h>
#include<iostream>
//#include<chrono>
#include"rBRIEF.cuh"
#include <fstream>

/*=============*/
#define PRINTSTATS
/*=============*/

int main(int argc, char const *argv[]) {
  //
  // //CPU=========================================================================
  //
  // // 1) Initialized arguments
  // int numPatch = 10;
  // int patchDim = 10;
  // float* patchArray = (float*) malloc(sizeof(float) * numPatch * patchDim * patchDim);
  // bool* binVectorArray = (bool*) malloc(sizeof(bool) * numPatch * 256);
  // for (int i = 0; i < numPatch * patchDim * patchDim; i++) {
  //   patchArray[i] = static_cast <float> (rand()) / static_cast <float> (255.0);
  // }
  // extern int cpu_precompute_BRIEF_pattern[256*4];
  // int* pattern = cpu_precompute_BRIEF_pattern;
  //
  // // 2) Run cpu reference
  // auto t1 = std::chrono::high_resolution_clock::now();
  // cpu_oBRIEF(numPatch, patchDim, patchArray, binVectorArray, pattern);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  //
  // #ifdef PRINTSTATS
  // std::cout << "CPU reference: " << std::endl;
  // printMatrix<bool*>(binVectorArray, numPatch, 256);
  // std::cout << "CPU implementation takes: " << duration << " microseconds" <<std::endl;
  // #endif
  //
  // //GPU=========================================================================
  //
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
  int P = 43; // number of images in the patches array
  int K = 128;// number of keypoints per patches
  float4 * gpu_patches;
  int4* gpu_pattern;
  int4* train_bin_vec;
  cudaMallocManaged(&gpu_patches, sizeof(float4) * 24 * P);
  cudaMallocManaged(&gpu_pattern, sizeof(int4) * 256);
  cudaMallocManaged(&train_bin_vec, sizeof(int4) * K);

  std::fstream myfile("./patches.txt", std::ios_base::in);
  float a;

  // 5) Get the values of the patches
  for (int i = 0; i < P * 24; i++) {
    myfile >> a;
    float x = a;
    myfile >> a;
    float y = a;
    myfile >> a;
    float z = a;
    myfile >> a;
    float w = a;
    gpu_patches[i] = make_float4(x,y,z,w);
  }

  // 6) Get the values of the pattern
  for (int i = 0; i < 256; i++) {
    int x = cpu_precompute_BRIEF_pattern[i*4 + 0];
    int y = cpu_precompute_BRIEF_pattern[i*4 + 1];
    int z = cpu_precompute_BRIEF_pattern[i*4 + 2];
    int w = cpu_precompute_BRIEF_pattern[i*4 + 3];
    gpu_pattern[i] = make_int4(x,y,z,w);
  }

  // 7) Get the values of the trained binary vector
  for (int i = 0; i < 32; i++) {
    int x = 460;
    int y = 146;
    int z = 241;
    int w = 124;
    train_bin_vec[i] = make_int4(x, y, z, w);
  }

  // 8) Run gpu
  gpu_rBRIEF(gpu_patches, gpu_pattern, train_bin_vec, K, P);
}

