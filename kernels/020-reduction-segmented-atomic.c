#include "common.h"

// Multi-workgroup sum reduction for arbitrary length inputs. The input is cut
// into segments of 2 * BLOCK_DIM elements, each workgroup reduces its segment
// in LDS and atomically adds its partial sum to *output.
//
// *output must be zeroed before launch.
// Workgroup: {BLOCK_DIM, 1, 1}
// Grid:      {ceil(n / (2 * BLOCK_DIM)) * BLOCK_DIM, 1, 1}

#define BLOCK_DIM 256

KERNEL
reduction_segmented_atomic(float* output, const float* input, unsigned int n) {
  SHARED float input_s[BLOCK_DIM];

  const unsigned int segment = 2 * BLOCK_DIM * group_id_x();
  const unsigned int t = local_id_x();
  const unsigned int i = segment + t;

  float sum = 0.0f;
  if (i < n) sum += input[i];
  if (i + BLOCK_DIM < n) sum += input[i + BLOCK_DIM];
  input_s[t] = sum;

  for (unsigned int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
    barrier();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }
  if (t == 0) {
    atomic_add(output, input_s[0]);
  }
}
