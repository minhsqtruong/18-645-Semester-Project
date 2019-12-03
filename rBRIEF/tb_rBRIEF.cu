#include<stdlib.h>
#include<iostream>
//#include<chrono>
#include"rBRIEF.cuh"
#include <fstream>

/*=============*/
#define PRINTSTATS
/*=============*/

int main(int argc, char const *argv[]) {

  int WPB = argv[1];
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
  int K = 96;  // number of pixel per patch
  int P = 128; // number of patches in one image
  int I = 1000;// number of images in the array
  int S = 32;  // number of bits in one binary vector
  float4 * gpu_patches;
  float  * raw_patches;
  int4* gpu_pattern;
  int4* train_bin_vec;
  raw_patches = (float *) malloc(sizeof(float) * K * P);
  cudaMallocManaged(&gpu_patches, sizeof(float4) * (K / 4) * P * I);
  cudaMallocManaged(&gpu_pattern, sizeof(int4) * 256);
  cudaMallocManaged(&train_bin_vec, sizeof(int4) * (S / 4) * P);

  std::fstream myfile("./141patches.txt", std::ios_base::in);
  float a;

  // 5) Get the values of the patches
  for (int pixel = 0; pixel < K * P; pixel++) {
    myfile >> a;
    raw_patches[pixel] = a;
  }
  for (int img = 0; img < I; img++) {
    for (int pixel= 0; pixel< (K * P) / 4; pixel++) {
      float x = raw_patches[pixel * 4 + 0];
      float y = raw_patches[pixel * 4 + 1];
      float z = raw_patches[pixel * 4 + 2];
      float w = raw_patches[pixel * 4 + 3];
      gpu_patches[img * (K * P) / 4 + pixel] = make_float4(x,y,z,w);
    }
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
  for (int run = 0; run < 1; run++) {
    gpu_rBRIEF(gpu_patches, gpu_pattern, train_bin_vec, K, P, I, WPB);
    cudaDeviceSynchronize();
  }
}
