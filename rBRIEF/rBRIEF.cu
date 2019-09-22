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
gpu_oBRIEF_kernel();

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
  int num_block  = 10;
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

  int ax, ay, bx, by;
  int rotated_ax, rotated_ay, rotated_bx, rotated_by;
  int Ia, Ib;
  int center_column;
  int kp = kp_x + kp_y * image_dim;
  float * patch = new float[patch_dim * patch_dim];
  for (int i = 0; i < patch_dim; i++) {
    center_column = kp + (i - patch_dim / 2) * image_dim;
    for (int j = 0; j < patch_dim; j++) {
      patch[j + i * patch_dim] = image[center_column + (j - patch_dim)];
    }
  }

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
