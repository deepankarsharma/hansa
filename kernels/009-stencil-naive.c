#include "common.h"

// Naive 3D seven-point stencil on an n x n x n grid. Each work-item computes
// one interior output point straight from global memory; boundary points are
// left untouched.
//
// Workgroup: e.g. {8, 8, 8}
// Grid:      {n, n, n}

KERNEL
stencil_naive(float* out, const float* in, int n, float c0, float c1, float c2,
              float c3, float c4, float c5, float c6) {
  const int i = global_id_z();
  const int j = global_id_y();
  const int k = global_id_x();

  if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
    const int idx = i * n * n + j * n + k;
    out[idx] = c0 * in[idx] + c1 * in[idx - 1] +
               c2 * in[idx + 1] +                            // x neighbours
               c3 * in[idx - n] + c4 * in[idx + n] +         // y neighbours
               c5 * in[idx - n * n] + c6 * in[idx + n * n];  // z neighbours
  }
}
