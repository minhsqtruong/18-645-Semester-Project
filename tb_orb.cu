#include<iostream>
#include <fstream>
#include "orb.cuh"
#include <stdlib.h>
#include "./oFAST/image.h"
int main(int argc, char const *argv[]) {

    // Begin oFAST
    int k = 1;      // Number of images
    int arr_size = 300;      // 300 is good for threshold 50
    int threshold = 35;  // 35 is default
    int row = 750;
    int col = 878;

    // Storing image into bigger image array
    uint8_t* img_sample_big;
    cudaMallocManaged(&img_sample_big, k*row*col);

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < col*row; j++)
        {
            img_sample_big[i*col*row+j] = img_sample[j];
        }
    }



    // Memory allocation for output coordinates data
    int *x_data;
    int *y_data;
    cudaMallocManaged(&x_data, k * arr_size * sizeof(int));
    cudaMallocManaged(&y_data, k * arr_size * sizeof(int));
    
    // Output data
    float4 *data_out;
    cudaMallocManaged(&data_out, k * 24 * arr_size * sizeof(float4));

    gpu_oFAST(img_sample_big, row, col, threshold, data_out, arr_size, k, x_data, y_data);

    

    
    // Printing float4
    for (int i = 0; i < 24 * arr_size * k; i++)
    {
        // printf("%f %f %f %f\n", data_out[i].x, data_out[i].y, data_out[i].z, data_out[i].w);
    }


    
    // End oFAST

    int WPB = atoi(argv[1]);

    int K = 96;  // number of pixel per patch
    int P = 128; // number of patches in one image (aka number of keypoints)
    int I = 1;// number of images in the array
    //int S = 32;  // number of bits in one binary vector
    float4 * gpu_patches;
    //float  * raw_patches;
    int4* gpu_pattern;
    int4* train_bin_vec;
    int * gpu_output;
    //raw_patches = (float *) malloc(sizeof(float) * K * P);
    cudaMallocManaged(&gpu_patches, sizeof(float4) * (K / 4) * P * I);
    cudaMallocManaged(&gpu_pattern, sizeof(int4) * 256);
    cudaMallocManaged(&train_bin_vec, sizeof(int4) * (P/4));
    cudaMallocManaged(&gpu_output, sizeof(int) * P * I);

  // Obsolete code
    // std::fstream myfile("./rBRIEF/141patches.txt", std::ios_base::in);
    // float a;

    // // 5) Get the values of the patches
    // for (int pixel = 0; pixel < K * P; pixel++) {
        // myfile >> a;
        // raw_patches[pixel] = a;
    // }


    // for (int img = 0; img < I; img++) {
        // for (int pixel= 0; pixel< (K * P) / 4; pixel++) {
          // float x = gpu_data[pixel * 4 + 0];
          // float y = gpu_data[pixel * 4 + 1];
          // float z = gpu_data[pixel * 4 + 2];
          // float w = gpu_data[pixel * 4 + 3];
          // gpu_patches[img * (K * P) / 4 + pixel] = make_float4(x,y,z,w);
        // }
    // }
    

    // Getting only the first 128 keypoints' patches from data_out (there are 300 originally)
    for (int i = 0; i < P*I*K/4; i++)
    {
        gpu_patches[i] = data_out[i];
    }
    
    for (int i = 0; i < P*I*K/4; i++)
    {
        // printf("%f %f %f %f\n", gpu_patches[i].x, gpu_patches[i].y, gpu_patches[i].z, gpu_patches[i].w);
    }
    // TODO: Segmentation error

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

