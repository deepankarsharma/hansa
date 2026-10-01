#include "common.h"

// Single-workgroup sum reduction with a convergent access pattern. The stride
// starts at the workgroup size and halves each step; the active work-items are
// always the contiguous range [0, stride), so whole wavefronts retire at once
// and memory accesses are coalesced.
//
// input has exactly 2 * workgroup_size elements and is overwritten.
// Workgroup: {B, 1, 1} with B a power of two (<= 1024)
// Grid:      {B, 1, 1} (a single workgroup)

KERNEL
reduction_convergent(float* output, float* input) {
  const unsigned int i = local_id_x();
  for (unsigned int stride = group_size_x(); stride >= 1; stride /= 2) {
    if (i < stride) {
      input[i] += input[i + stride];
    }
    barrier();
  }
  if (i == 0) {
    *output = input[0];
  }
}
