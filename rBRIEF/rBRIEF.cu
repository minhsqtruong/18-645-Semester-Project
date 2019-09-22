#include "stdio.h"

// write implementation of rBRIEF module here


void cpu_oBRIEF(float sin_theta,
                float cos_theta,
                float* patch,
                int patch_dim,
                int * pattern,
                bool * binary_feature
                )
{

  int ax, ay, bx, by;
  int rotated_ax, rotated_ay, rotated_bx, rotated_by;
  int Ia, Ib;

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

// test pipeline integration
void pipeline_print_rBRIEF(){ printf("rBRIEF Module active!\n");};
