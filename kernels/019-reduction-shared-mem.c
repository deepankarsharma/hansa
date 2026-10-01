#include "common.h"

// Single-workgroup sum reduction in LDS. The first step is folded into the
// load from global memory, after which all partial sums live in LDS: global
// memory is read exactly once and never written except for the result, and
// the input is no longer modified.
//
// input has exactly 2 * BLOCK_DIM elements.
// Workgroup: {BLOCK_DIM, 1, 1}
// Grid:      {BLOCK_DIM, 1, 1} (a single workgroup)

#define BLOCK_DIM 256

KERNEL
reduction_shared_mem(float* output, const float* input) {
  SHARED float input_s[BLOCK_DIM];

  const unsigned int t = local_id_x();
  input_s[t] = input[t] + input[t + BLOCK_DIM];
  for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
    barrier();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    *output = input_s[0];
  }
}
