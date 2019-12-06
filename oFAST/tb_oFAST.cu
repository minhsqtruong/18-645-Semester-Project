#include "oFAST.cuh"
#include "image.h"

int main(int argc, char const *argv[]) {
    cudaProfilerStart();

    int k = 1;
    int arr_size = 300;      // 300 is good for threshold 50
    //int threshold = atoi(argv[1]);  // 35 is default
    int threshold = 35;  // 35 is default
    
    dim3 block(32, 8);

    int height = 878;
    int width = 750;

    dim3 grid;
    grid.x = divUp(height - 6, block.x);
    grid.y = divUp(width - 6, block.y);
    //printf("%d", sizeof(img_sample));
    uint8_t* img_sample_big;
    cudaMallocManaged(&img_sample_big, k*height*width);
    //printf("%li\n", sizeof(img_sample));
    //printf("%li\n", sizeof(img_sample_big));

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < width*height; j++)
        {
            img_sample_big[i*width*height+j] = img_sample[j];
        }
    }
    for (int i = 0; i < k*height*width; i++)
    {
        //printf("%d ",img_sample_big[i]);
    }


    // Copying sample image to device
    uint8_t* img_d;
    cudaMalloc(&img_d, height*width*sizeof(uint8_t));
    cudaMemcpy(img_d, img_sample, height*width*sizeof(uint8_t), cudaMemcpyHostToDevice);



    // Memory allocation for output brightness data
    float *gpu_data;
    cudaMallocManaged(&gpu_data, k * 100 * arr_size * sizeof(float));

    // Memory allocation for output coordinates data
    uint8_t *x_data;
    uint8_t *y_data;
    cudaMallocManaged(&x_data, k * arr_size * sizeof(uint8_t));
    cudaMallocManaged(&y_data, k * arr_size * sizeof(uint8_t));

    // Memory allocation for c_table
    uint8_t *ctable_gpu;
    cudaMallocManaged(&ctable_gpu, 8129);
    for (int i = 0; i < 8129; i++)
    {
        ctable_gpu[i] = c_table[i];
    }

    ///////////////////////////////////////////////////////
    // Most important line of the file/////////////////////
    // Launch the kernel//////////////////////////////////////////
    calcKeyPoints<<<grid, block, 8129>>>(img_sample_big, width, height, threshold, gpu_data, arr_size, k, x_data, y_data, ctable_gpu);
    //////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

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
        // printf("%f %f %f %f\n", data_out[i].x, data_out[i].y, data_out[i].z, data_out[i].w);
    }

    // Printing coordinates
    for (i = 0; i < arr_size; i++)
    {
        printf("%d %d\n", x_data[i], y_data[i]);
    }

    // Free device memory

    cudaFree(img_d);
    cudaFree(img_sample);
    cudaFree(gpu_data);
    cudaFree(data_out);
    cudaFree(img_sample_big);
    cudaFree(x_data);
    cudaFree(y_data);
    cudaFree(ctable_gpu);
    cudaProfilerStop();
    cudaGetLastError();

    cudaDeviceSynchronize();

    return 0;
}
