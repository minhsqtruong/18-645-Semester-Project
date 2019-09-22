#include"rBRIEF.cuh"
#include<stdio.h>
int main(int argc, char const *argv[]) {

  // Initialization
  float sin_theta = 0.9;
  float cos_theta = 0.3;
  float * image;
  cudaErrorCheck(cudaMallocManaged(&image, 1000 * sizeof(float)));
  for (int i = 0; i < 1000; i++) {
    image[i] = (float) i;
  }

  int kp_x = 50;
  int kp_y = 20;
  int image_dim = 100;
  int patch_dim = 10;
  int * pattern;
  cudaErrorCheck(cudaMallocManaged(&pattern, 256 * 4 * sizeof(int)));
  cudaErrorCheck(cudaMemcpy(&pattern,
                           &cpu_precompute_BRIEF_pattern,
                           256 * 4 * sizeof(int),
                           cudaMemcpyDefault));

  bool cpu_binary_feature[256];
  bool * gpu_binary_feature;
  cudaErrorCheck(cudaMallocManaged(&gpu_binary_feature, 256 * sizeof(bool)));

  // Test Start
  cpu_oBRIEF( sin_theta,
              cos_theta,
              kp_x,
              kp_y,
              image,
              image_dim,
              patch_dim,
              pattern,
              cpu_binary_feature
            );

  gpu_oBRIEF( sin_theta,
              cos_theta,
              kp_x,
              kp_y,
              image,
              image_dim,
              patch_dim,
              pattern,
              gpu_binary_feature
            );

  validate_gpu_result(cpu_binary_feature, gpu_binary_feature);

  // test pipeline integration
  //pipeline_print_rBRIEF();
  return 0;
}
