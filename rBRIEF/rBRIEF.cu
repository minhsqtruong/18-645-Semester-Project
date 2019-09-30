#include "stdio.h"
/*==============================================================================

==============================================================================*/
void validate_gpu_result(bool * cpu_binary_feature,
                         bool * gpu_binary_feature)
{
  for (int i = 0; i < 256; i++){
    if (cpu_binary_feature[i] != gpu_binary_feature[i])
      printf("Error cpu[%d] = %d while gpu[%d] = %d\n", i, cpu_binary_feature[i]
                                                      , i, gpu_binary_feature[i]);
  }
};
/*==============================================================================

==============================================================================*/
void __global__
gpu_oBRIEF_kernel(float sin_theta,
                  float cos_theta,
                  int kp_x, //TODO change to array
                  int kp_y, //TODO change to array
                  float * image,
                  int image_dim,
                  int image_size, // = image_dim ^ 2
                  int patch_dim,
                  int patch_size, // = patch_dim ^ 2
                  int * pattern,
                  bool * binary_feature)
{
  //INITIATIONS AND ALLOCATIONS
  extern __shared__ float shared_mem[];
  float* shared_image = &shared_mem[0];
  float* shared_patch = &shared_mem[image_size];
  int local_id = threadIdx.x;
  int stride = blockDim.x;
  int kp;
  int center_column;
  int ax, ay, bx, by;
  int rotated_ax, rotated_ay, rotated_bx, rotated_by;
  int Ia, Ib;
  int i = local_id / patch_dim;
  int j = local_id % patch_dim;

  // LOAD IMAGE INTO SHARED MEMORY TODO look into unrolling this loop
  for (int i = local_id; i < image_size; i += stride) {
    shared_image[i] = image[i];
  }
  __syncthreads();

  // ALLOCATE PATCH OF THE SPECIFIED KEYPOINT
  // TODO change this to loop
  // TODO look into prefetching the next patch
  kp = kp_x + kp_y * image_dim;
  if (local_id < patch_size) {
        center_column = kp + (i - patch_dim / 2) * image_dim;
        shared_patch[j + i * patch_dim]
        = shared_image[center_column + (j - patch_dim)];
  }
  __syncthreads();

  // SAMPLE THE PATCH BASED ON PATTERN AND RETURN THE BINARY FEATURE
  ax = pattern[4*local_id];
  ay = pattern[4*local_id+1];
  bx = pattern[4*local_id+2];
  by = pattern[4*local_id+3];

  rotated_ax = (int) (cos_theta * ax - sin_theta * ay);
  rotated_ay = (int) (sin_theta * ay + cos_theta * ay);
  rotated_bx = (int) (cos_theta * bx - sin_theta * by);
  rotated_by = (int) (sin_theta * by + cos_theta * by);

  Ia = shared_patch[rotated_ax + patch_dim * rotated_ay];
  Ib = shared_patch[rotated_bx + patch_dim * rotated_by];

  binary_feature[local_id] = Ia > Ib;

};

void gpu_oBRIEF(float sin_theta,
                float cos_theta,
                int kp_x,
                int kp_y,
                float * image,
                int image_dim,
                int patch_dim,
                int * pattern,
                bool * binary_feature
                )
{
  int num_thread = 256;
  int num_kp_per_block = 10;
  int num_block  = 1; // TODO replace by actual computation later.
  int image_size = image_dim * image_dim;
  int patch_size = patch_dim * patch_dim;
  int shared_mem_size = sizeof(float) * (image_size + patch_size);
  gpu_oBRIEF_kernel<<<num_block, num_thread, shared_mem_size>>>
  (sin_theta,
   cos_theta,
   kp_x, //TODO change to array
   kp_y, //TODO change to array
   image,
   image_dim,
   image_size, // = image_dim ^ 2
   patch_dim,
   patch_size, // = patch_dim ^ 2
   pattern,
   binary_feature);
  cudaDeviceSynchronize();
};
/*==============================================================================

==============================================================================*/
void cpu_oBRIEF(float sin_theta,
                float cos_theta,
                int kp_x,
                int kp_y,
                float * image,
                int image_dim,
                int patch_dim,
                int * pattern,
                bool * binary_feature
                )
{

  // INITIATIONS AND ALLOCATIONS
  int ax, ay, bx, by;
  int rotated_ax, rotated_ay, rotated_bx, rotated_by;
  int Ia, Ib;
  int center_column;
  int kp = kp_x + kp_y * image_dim; // find the keypoint index in 1D format
  float * patch = new float[patch_dim * patch_dim]; // allocate space for patch


  // ALLOCATE PATCH OF THE SPECIFIED KEYPOINT
  // find center point of every collumn
  for (int i = 0; i < patch_dim; i++) {
    center_column = kp + (i - patch_dim / 2) * image_dim;
    // iterate through each column
    for (int j = 0; j < patch_dim; j++) {
      patch[j + i * patch_dim] = image[center_column + (j - patch_dim)];
    }
  }

  // SAMPLE THE PATCH BASED ON PATTERN AND RETURN THE BINARY FEATURE
  for (int i = 0; i < 256; ++i) {
    ax = pattern[4*i];
    ay = pattern[4*i+1];
    bx = pattern[4*i+2];
    by = pattern[4*i+3];

    rotated_ax = (int) (cos_theta * ax - sin_theta * ay);
    rotated_ay = (int) (sin_theta * ay + cos_theta * ay);
    rotated_bx = (int) (cos_theta * bx - sin_theta * by);
    rotated_by = (int) (sin_theta * by + cos_theta * by);

    Ia = patch[rotated_ax + patch_dim * rotated_ay];
    Ib = patch[rotated_bx + patch_dim * rotated_by];

    binary_feature[i] = Ia > Ib;
  }
};
/*==============================================================================

==============================================================================*/
void pipeline_print_rBRIEF(){ printf("rBRIEF Module active!\n");};
