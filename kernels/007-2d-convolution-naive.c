#include "common.h"

// Naive 2D convolution: each work-item computes one output pixel by reading
// its (2r+1) x (2r+1) neighbourhood straight from global memory. Out of bounds
// input pixels ("ghost cells") are treated as 0.
//
// filter is a (2r+1) x (2r+1) row-major matrix.
// Workgroup: e.g. {16, 16, 1}
// Grid:      {width, height, 1}

KERNEL
convolution_2d_naive(float* out, const float* in, const float* filter, int r,
                     int width, int height) {
  const int col = global_id_x();
  const int row = global_id_y();
  if (col >= width || row >= height) return;

  const int filter_dim = 2 * r + 1;
  float sum = 0.0f;
  for (int f_row = 0; f_row < filter_dim; f_row++) {
    for (int f_col = 0; f_col < filter_dim; f_col++) {
      const int in_row = row - r + f_row;
      const int in_col = col - r + f_col;
      if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
        sum += filter[f_row * filter_dim + f_col] * in[in_row * width + in_col];
      }
    }
  }
  out[row * width + col] = sum;
}
