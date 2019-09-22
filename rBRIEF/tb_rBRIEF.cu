#include"rBRIEF.cuh"
#include<stdio.h>
int main(int argc, char const *argv[]) {

  // write test code for rBRIEF module here
  float sin_theta = 0.9;
  float cos_theta = 0.3;
  float patch [100];
  int patch_dim = 10;
  int * pattern = cpu_precompute_BRIEF_pattern;
  bool binary_feature[256];

  // initialize patch
  for (int i = 0; i < 100; i++) {
    patch[i] = (float) i;
  }
  cpu_oBRIEF(sin_theta,
            cos_theta,
            patch,
            patch_dim,
            pattern,
            binary_feature);

  // print binary_feature
  for (int i = 0; i < 256; i++) {
    printf("%d\n", binary_feature[i]);
  }
  // test pipeline integration
  //pipeline_print_rBRIEF();
  return 0;
}
