#include <stdio.h>
#include <stdint.h>
#include "ctable.h"
#include "image.h"
#include <math.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

__device__ unsigned int g_counter = 0;


// This function returns
// 1 if v is greater than x + th
// 2 if v is less than x - th
// 0 if v is between x + th and x - th
__device__ __forceinline__ int diffType(const int v, const int x, const int th)
{
    const int diff = x - v;
    return static_cast<int>(diff < -th) + (static_cast<int>(diff > th) << 1);
}


// Integer division with result round up
__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}


// mask1/2 light/dark
__device__ void calcMask(const int C[4], const int v, const int th, int& mask1, int& mask2)
{
    mask1 = 0;  // only cares about bright one
    mask2 = 0;  // only cares about dark

    int d1, d2;



    d1 = diffType(v, C[0] & 0xff, th);
    d2 = diffType(v, C[2] & 0xff, th);

    if ((d1 | d2) == 0) // if both sides are between the thresholds
        return;

    mask1 |= (d1 & 1) << 0;

    // because we're shifting 2'b10 left, we need to shift one back right
    mask2 |= ((d1 & 2) >> 1) << 0;

    mask1 |= (d2 & 1) << 8;
    mask2 |= ((d2 & 2) >> 1) << 8;



    d1 = diffType(v, C[1] & 0xff, th);
    d2 = diffType(v, C[3] & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 4;
    mask2 |= ((d1 & 2) >> 1) << 4;

    mask1 |= (d2 & 1) << 12;
    mask2 |= ((d2 & 2) >> 1) << 12;

    // end of four corners

    d1 = diffType(v, (C[0] >> (2 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (2 * 8)) & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 2;
    mask2 |= ((d1 & 2) >> 1) << 2;

    mask1 |= (d2 & 1) << 10;
    mask2 |= ((d2 & 2) >> 1) << 10;



    d1 = diffType(v, (C[1] >> (2 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (2 * 8)) & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 6;
    mask2 |= ((d1 & 2) >> 1) << 6;

    mask1 |= (d2 & 1) << 14;
    mask2 |= ((d2 & 2) >> 1) << 14;



    d1 = diffType(v, (C[0] >> (1 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (1 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 1;
    mask2 |= ((d1 & 2) >> 1) << 1;

    mask1 |= (d2 & 1) << 9;
    mask2 |= ((d2 & 2) >> 1) << 9;



    d1 = diffType(v, (C[0] >> (3 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (3 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 3;
    mask2 |= ((d1 & 2) >> 1) << 3;

    mask1 |= (d2 & 1) << 11;
    mask2 |= ((d2 & 2) >> 1) << 11;



    d1 = diffType(v, (C[1] >> (1 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (1 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 5;
    mask2 |= ((d1 & 2) >> 1) << 5;

    mask1 |= (d2 & 1) << 13;
    mask2 |= ((d2 & 2) >> 1) << 13;



    d1 = diffType(v, (C[1] >> (3 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (3 * 8)) & 0xff, th);

    mask1 |= (d1 & 1) << 7;
    mask2 |= ((d1 & 2) >> 1) << 7;

    mask1 |= (d2 & 1) << 15;
    mask2 |= ((d2 & 2) >> 1) << 15;
}

// 1 -> v > x + th
// 2 -> v < x - th
// 0 -> not a keypoint

// popc counts the number of 1's
__device__ __forceinline__ bool isKeyPoint(int mask1, int mask2, uint8_t *shared_table)
{
    return (__popc(mask1) > 8 && (shared_table[(mask1 >> 3) - 63] & (1 << (mask1 & 7)))) ||
           (__popc(mask2) > 8 && (shared_table[(mask2 >> 3) - 63] & (1 << (mask2 & 7))));
}


// This is my kernel
__global__ void calcKeyPoints(uint8_t* image, int rows, int cols, int threshold, float *data, int arr_size, int k, uint8_t *x_data, uint8_t *y_data, uint8_t *ctable_gpu)
{
    extern __shared__ uint8_t shared_table[];
    for (int ind = threadIdx.x; ind < 8129; ind+=blockDim.x)
    {
        shared_table[ind] = ctable_gpu[ind];
    }
    const int j = threadIdx.x + blockIdx.x * blockDim.x + 10;
    const int i = threadIdx.y + blockIdx.y * blockDim.y + 10;

    for (int a = 0; a < k; a++)
    {
        int next = a * rows * cols;
        uint8_t* img = image + next;

        if (i < rows - 10 && j < cols - 10)
        {
            int i_minus_three = cols*(i-3);
            int j_minus_three = (j-3);
            int j_plus_three = (j+3);
            int i_plus_three = cols * (i + 3);
            int v;
            int C[4] = {0,0,0,0};
            C[2] |= static_cast<uint8_t>(img[i_minus_three + (j - 1)]) << 8;

            C[2] |= static_cast<uint8_t>(img[i_minus_three + (j)]);

            C[1] |= static_cast<uint8_t>(img[i_minus_three + (j + 1)]) << (3 * 8);

            C[2] |= static_cast<uint8_t>(img[cols*(i - 2) + (j - 2)]) << (2 * 8);
            C[1] |= static_cast<uint8_t>(img[cols*(i - 2) + (j + 2)]) << (2 * 8);

            C[2] |= static_cast<uint8_t>(img[cols*(i - 1) + j_minus_three]) << (3 * 8);
            C[1] |= static_cast<uint8_t>(img[cols*(i - 1) + j_plus_three]) << 8;

            C[3] |= static_cast<uint8_t>(img[cols * (i) + j_minus_three]);
            v     = static_cast<uint8_t>(img[cols * (i) + (j)]);
            C[1] |= static_cast<uint8_t>(img[cols * (i) + j_plus_three]);
            // Checking both sides
            int d1 = diffType(v, C[1] & 0xff, threshold);
            int d2 = diffType(v, C[3] & 0xff, threshold);

            if ((d1 | d2) == 0)
            {
                return;
            }

            C[3] |= static_cast<uint8_t>(img[cols * (i + 1) + j_minus_three]) << 8;
            C[0] |= static_cast<uint8_t>(img[cols * (i + 1) + j_plus_three]) << (3 * 8);

            C[3] |= static_cast<uint8_t>(img[cols * (i + 2) + (j - 2)]) << (2 * 8);
            C[0] |= static_cast<uint8_t>(img[cols * (i + 2) + (j + 2)]) << (2 * 8);

            C[3] |= static_cast<uint8_t>(img[i_plus_three + (j - 1)]) << (3 * 8);
            C[0] |= static_cast<uint8_t>(img[i_plus_three + (j)]);
            C[0] |= static_cast<uint8_t>(img[i_plus_three + (j + 1)]) << 8;

            int mask1 = 0;
            int mask2 = 0;

            calcMask(C, v, threshold, mask1, mask2);



            if (isKeyPoint(mask1, mask2, shared_table))
            {
                const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));

                if (ind < arr_size)
                {
                    x_data[ind] = i;
                    y_data[ind] = j;
                    #pragma unroll
                    for (int b = 0; b < 100; b++)
                    {
                        // Getting the patch
                        data[(ind*100)+b] = static_cast<float>(img[cols*(i+4-(b/10))+(j+(-4+(b%10)))]);

                    }
                }
            }

        }
    }

}


int main(int argc, char *argv[])
{
    cudaProfilerStart();

    int k = 1;
    int arr_size = 300;      // 300 is good for threshold 50
    //int threshold = atoi(argv[1]);  // 35 is default
    int threshold = 50;  // 35 is default
    
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
