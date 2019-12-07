#include "oFAST.cuh"
#include "image.h"

int main(int argc, char const *argv[]) {
    cudaProfilerStart();

    int k = 1000;
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

    // Memory allocation for output brightness data
    float *gpu_data;
    cudaMallocManaged(&gpu_data, k * 100 * arr_size * sizeof(float));

    // Memory allocation for output coordinates data
    int *x_data;
    int *y_data;
    cudaMallocManaged(&x_data, k * arr_size * sizeof(int));
    cudaMallocManaged(&y_data, k * arr_size * sizeof(int));

    gpu_oFAST(img_sample_big, row, col, threshold, gpu_data, arr_size, k, x_data, y_data);

    cudaDeviceSynchronize();

    // Putting in float4
    float4 *data_out;
    cudaMallocManaged(&data_out, k * 24 * arr_size * sizeof(float4));

    int i = 0;
    int j = 0;
    int c = 0;

    while (c < arr_size * k)
    {
        while (j < 24)
        {
            data_out[24*c+j] = make_float4(gpu_data[100*c+i], gpu_data[100*c+i+1], gpu_data[100*c+i+2], gpu_data[100*c+i+3]);
            i = i + 4;
            j++;
        }
        i = 0;
        j = 0;
        c++;
    }

    // Printing float4
    for (i = 0; i < 24 * arr_size * k; i++)
    {
        //printf("%f %f %f %f\n", data_out[i].x, data_out[i].y, data_out[i].z, data_out[i].w);
    }

    // Printing coordinates
    for (i = 0; i < arr_size; i++)
    {
        //printf("%d %d\n", x_data[i], y_data[i]);
    }

    // Free device memory

    //cudaFree(img_d);
    cudaFree(img_sample);
    cudaFree(gpu_data);
    cudaFree(data_out);
    cudaFree(img_sample_big);
    cudaFree(x_data);
    cudaFree(y_data);
    cudaProfilerStop();
    cudaGetLastError();

    cudaDeviceSynchronize();

    return 0;
}
