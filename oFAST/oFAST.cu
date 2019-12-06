#include "oFAST.cuh"

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
__global__ void calcKeyPoints(uint8_t* image, int rows, int cols, int threshold, float *data, int arr_size, int k, int *x_data, int *y_data, uint8_t *ctable_gpu)
{
    
    extern __shared__ uint8_t shared_table[];
    for (int ind = threadIdx.x; ind < 8129; ind+=blockDim.x)
    {
        shared_table[ind] = ctable_gpu[ind];
    }
    const int j = threadIdx.x + blockIdx.x * blockDim.x + 10;
    const int i = threadIdx.y + blockIdx.y * blockDim.y + 10;
    //printf("%d %d\n", i, j);
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
                unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));
                //printf("%d\n", ind);
                //printf("%d %d\n", i, j);
                if (ind < k * arr_size)
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

void gpu_oFAST(uint8_t* image, int rows, int cols, int threshold, float *data, int arr_size, int k, int *x_data, int *y_data)
{
    dim3 block(32, 8);
    dim3 grid;
    grid.x = divUp(rows - 6, block.x);
    grid.y = divUp(cols - 6, block.y);
    // Memory allocation for c_table
    uint8_t *ctable_gpu;
    cudaMallocManaged(&ctable_gpu, 8129);
    for (int i = 0; i < 8129; i++)
    {
        ctable_gpu[i] = c_table[i];
    }

    calcKeyPoints<<<grid, block, 8129>>>(image, rows, cols, threshold, data, arr_size, k, x_data, y_data, ctable_gpu);
}


// test pipeline integration
void pipeline_print_oFAST(){ printf("oFAST Module active!\n");};
