#include<iostream>
#include <fstream>
#include "orb.cuh"
#include <stdlib.h>
int main(int argc, char const *argv[]) {

  int WPB = atoi(argv[1]);

  int K = 96;  // number of pixel per patch
  int P = 128; // number of patches in one image
  int I = 1000;// number of images in the array
  //int S = 32;  // number of bits in one binary vector
  float4 * gpu_patches;
  float  * raw_patches;
  int4* gpu_pattern;
  int4* train_bin_vec;
  int * gpu_output;
  raw_patches = (float *) malloc(sizeof(float) * K * P);
  cudaMallocManaged(&gpu_patches, sizeof(float4) * (K / 4) * P * I);
  cudaMallocManaged(&gpu_pattern, sizeof(int4) * 256);
  cudaMallocManaged(&train_bin_vec, sizeof(int4) * (P/4));
  cudaMallocManaged(&gpu_output, sizeof(int) * P * I);

  std::fstream myfile("./rBRIEF/141patches.txt", std::ios_base::in);
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

  //7) Get the values of the trained binary vector
  for (int i = 0; i < 32; i++) {
    int x = cpu_precomputed_BRIEF_binvec[i*4 + 0];
    int y = cpu_precomputed_BRIEF_binvec[i*4 + 1];
    int z = cpu_precomputed_BRIEF_binvec[i*4 + 2];
    int w = cpu_precomputed_BRIEF_binvec[i*4 + 3];
    train_bin_vec[i] = make_int4(x, y, z, w);
  }

  // 8) Run gpu
  gpu_rBRIEF(gpu_patches, gpu_output, gpu_pattern, train_bin_vec, K, P, I, WPB);
  cudaDeviceSynchronize();


  // test pipeline integration
  // pipeline_print_oFAST();
  // pipeline_print_rBRIEF();
  // pipeline_print_match();
  return 0;
}

