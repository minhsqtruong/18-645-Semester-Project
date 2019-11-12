#include <stdio.h>
#include <stdint.h>
#include "ctable.h"
#include "image.h"
#include <math.h>
#include <cuda.h>

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
__device__ void calcMask(const uint8_t C[4], const int v, const int th, int& mask1, int& mask2)
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
__device__ __forceinline__ bool isKeyPoint(int mask1, int mask2)
{
    return (__popc(mask1) > 8 && (c_table[(mask1 >> 3) - 63] & (1 << (mask1 & 7)))) ||
           (__popc(mask2) > 8 && (c_table[(mask2 >> 3) - 63] & (1 << (mask2 & 7))));
}

// This is my kernel
__global__ void calcKeyPoints(uint8_t* image, int rows, int cols, int threshold, int *arr_i, int *arr_j, float *data)
{
    
    const int j = threadIdx.x + blockIdx.x * blockDim.x + 10;
    const int i = threadIdx.y + blockIdx.y * blockDim.y + 10;
    //printf("i = %d, j = %d\n", i, j);
    
    
    
    if (i < rows - 10 && j < cols - 10)
    {
        
        
        int v;
        uint8_t C[4] = {0,0,0,0};
        //printf("pixel is %d\n", *(image+1));
        C[2] |= static_cast<uint8_t>(image[cols*(i - 3) + (j - 1)]) << 8;
        
        C[2] |= static_cast<uint8_t>(image[cols*(i - 3) + (j)]);
        
        C[1] |= static_cast<uint8_t>(image[cols*(i - 3) + (j + 1)]) << (3 * 8);
        
        C[2] |= static_cast<uint8_t>(image[cols*(i - 2) + (j - 2)]) << (2 * 8);
        C[1] |= static_cast<uint8_t>(image[cols*(i - 2) + (j + 2)]) << (2 * 8);

        C[2] |= static_cast<uint8_t>(image[cols*(i - 1) + (j - 3)]) << (3 * 8);
        C[1] |= static_cast<uint8_t>(image[cols*(i - 1) + (j + 3)]) << 8;

        C[3] |= static_cast<uint8_t>(image[cols * (i) + (j - 3)]);
        v     = static_cast<uint8_t>(image[cols * (i) + (j)]);
        C[1] |= static_cast<uint8_t>(image[cols * (i) + (j + 3)]);
        // Checking both sides
        int d1 = diffType(v, C[1] & 0xff, threshold);
        int d2 = diffType(v, C[3] & 0xff, threshold);
        //printf("d1 = %d\nd2 = %d\n", d1, d2);
        if ((d1 | d2) == 0)
        {
            return;
        }
        //printf("function called\n");
        C[3] |= static_cast<uint8_t>(image[cols * (i + 1) + (j - 3)]) << 8;
        C[0] |= static_cast<uint8_t>(image[cols * (i + 1) + (j + 3)]) << (3 * 8);

        C[3] |= static_cast<uint8_t>(image[cols * (i + 2) + (j - 2)]) << (2 * 8);
        C[0] |= static_cast<uint8_t>(image[cols * (i + 2) + (j + 2)]) << (2 * 8);

        C[3] |= static_cast<uint8_t>(image[cols * (i + 3) + (j - 1)]) << (3 * 8);
        C[0] |= static_cast<uint8_t>(image[cols * (i + 3) + (j)]);
        C[0] |= static_cast<uint8_t>(image[cols * (i + 3) + (j + 1)]) << 8;

        int mask1 = 0;
        int mask2 = 0;

        calcMask(C, v, threshold, mask1, mask2);
        
        
        
        if (isKeyPoint(mask1, mask2))
        {
            const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));
            //printf("ind = %d\n", ind);
            arr_i[ind] = i;
            arr_j[ind] = j;
            //printf("%f\n", static_cast<float>(image[cols*(5)+(j+(5))]));
            int k = 0;
            //int l = 0;
            //while (l < 10)
            //{
            while (k < 100)
            {
                //printf("%d ", k);
                //data[(ind*100)+l*10+k] = static_cast<float>(image[cols*(i+4-l)+(j+(-4+k))]);
                data[(ind*100)+k] = static_cast<float>(image[cols*(i+4-(k/10))+(j+(-4+(k%10)))]);
                //printf("%f,", data[ind*l*10+k]);
                k++;
            }
                //l++;
                //printf("test");
            //}
            //printf("\n");
            
            
        }
            
    }


}


int main()
{
    void* counter_ptr;
    cudaGetSymbolAddress(&counter_ptr, g_counter);
    
    int arr_size = 300;     // 300 is good for threshold 50
    
    dim3 block(32, 8);

    dim3 grid;
    grid.x = divUp(878 - 6, block.x);
    grid.y = divUp(750 - 6, block.y);
    
    //printf("block.x = %d\nblock.y = %d\n", block.x, block.y);

    cudaMemset(counter_ptr, 0, sizeof(unsigned int));
    
    // Copying sample image to device
    uint8_t* img_d;
    cudaMalloc(&img_d, 878*750*sizeof(uint8_t));
    cudaMemcpy(img_d, img_sample, 878*750*sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Mallocing indices in host
    int *arr_i, *arr_j;
    arr_i = (int*)malloc(arr_size*sizeof(int));
    arr_j = (int*)malloc(arr_size*sizeof(int));
    
    // Copying indices to device
    int *arr_i_d, *arr_j_d;
    cudaMalloc(&arr_i_d, arr_size*sizeof(int));
    cudaMalloc(&arr_j_d, arr_size*sizeof(int));
    cudaMemcpy(arr_i_d, arr_i, arr_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(arr_j_d, arr_j, arr_size*sizeof(int), cudaMemcpyHostToDevice);
    
    // Mallocing cpu_data in host
    float *cpu_data;
    cpu_data = (float*)malloc(100 * arr_size * sizeof(float));
    
    // Copying cpu_data to device
    float *gpu_data;
    cudaMalloc(&gpu_data, 100 * arr_size * sizeof(float));
    cudaMemcpy(gpu_data, cpu_data, 100 * arr_size * sizeof(float), cudaMemcpyHostToDevice);
    
    ///////////////////////////////////////////////////////
    // Most important line of the file/////////////////////
    // Launch the kernel//////////////////////////////////////////
    calcKeyPoints<<<grid, block>>>(img_d, 750, 878, 50, arr_i_d, arr_j_d, gpu_data);
    //////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    
    // Get the data back
    cudaMemcpy(cpu_data, gpu_data, 100 * arr_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Getting indices back from device
    cudaMemcpy(arr_i, arr_i_d, arr_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_j, arr_j_d, arr_size*sizeof(int), cudaMemcpyDeviceToHost);
    
    /*
    // Printing all patches of keypoints
    int i;
    for (i = 0; i < 100 * arr_size; i++)
    {
        printf("%f ", cpu_data[i]);
        if ((i+1)%100 == 0)
        {
            printf("\n");
        }
    }
    */
    
    // Putting in float4
    float4 *data_out;
    data_out = (*float4)malloc(25 * arr_size * sizeof(float4));
    
    int i = 0;
    int j = 0;
    while (i < 100 * arr_size)
    {
        data_out[j] = make_float4(cpu_data[i], cpu_data[i+1], cpu_data[i+2], cpu_data[i+3]);
        j++;
        i = i + 4;
    }
    
    
    // Printing float4
    for (i = 0; i < 25 * arr_size; i++)
    {
        printf("%f %f %f %f\n", data_out[i].x, data_out[i].y, data_out[i].z, data_out[i].w);
    }
    
    
    
    // Free device memory
    cudaFree(arr_i_d);
    cudaFree(arr_j_d);
    cudaFree(img_d);
    cudaFree(gpu_data);
    free(cpu_data);
    free(arr_i);
    free(arr_j);
    
    cudaGetLastError();

    cudaDeviceSynchronize();

    unsigned int count;
    cudaMemcpy(&count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //printf("count is %d\n", count);
    //printf("pixel is %d\n", img[878*(3 - 3) + (3 - 1)]);
    return count;
}