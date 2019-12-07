#include "oFAST.cuh"
#include "image.h"

int main(int argc, char const *argv[]) {
    cudaProfilerStart();

    int k = 1;
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

    cudaDeviceSynchronize();


    // Printing float4
    for (int i = 0; i < 24 * arr_size; i++)
    {
        // printf("%f %f %f %f\n", data_out[i].x, data_out[i].y, data_out[i].z, data_out[i].w);
    }

    // Printing coordinates
    for (int i = 0; i < arr_size; i++)
    {
        // printf("%d %d\n", x_data[i], y_data[i]);
    }

    // Free device memory

    //cudaFree(img_d);
    cudaFree(img_sample);
    cudaFree(data_out);
    cudaFree(img_sample_big);
    cudaFree(x_data);
    cudaFree(y_data);
    cudaProfilerStop();
    cudaGetLastError();

    cudaDeviceSynchronize();

    return 0;
}
