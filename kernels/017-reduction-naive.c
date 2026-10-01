#include "common.h"

// Naive single-workgroup sum reduction, performed in place in global memory.
// Work-item t owns element 2t; in each step the stride doubles and only
// work-items whose owned index is a multiple of 2*stride stay active. This
// leaves active work-items scattered across every wavefront (control
// divergence) and strided memory accesses.
//
// input has exactly 2 * workgroup_size elements and is overwritten.
// Workgroup: {B, 1, 1} with B a power of two (<= 1024)
// Grid:      {B, 1, 1} (a single workgroup)

KERNEL
reduction_naive(float* output, float* input) {
  const unsigned int t = local_id_x();
  const unsigned int i = 2 * t;
  for (unsigned int stride = 1; stride <= group_size_x(); stride *= 2) {
    if (t % stride == 0) {
      input[i] += input[i + stride];
    }
    barrier();
  }
  if (t == 0) {
    *output = input[0];
  }
}
