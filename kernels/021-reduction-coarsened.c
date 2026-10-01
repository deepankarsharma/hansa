#include "common.h"

// Multi-workgroup sum reduction with thread coarsening. Each work-item first
// serially sums COARSE_FACTOR * 2 elements (strided by BLOCK_DIM so the loads
// stay coalesced) before the tree reduction in LDS. Fewer workgroups means
// fewer barrier-synchronized tree steps and fewer atomics for the same input.
//
// *output must be zeroed before launch.
// Workgroup: {BLOCK_DIM, 1, 1}
// Grid:      {ceil(n / (COARSE_FACTOR * 2 * BLOCK_DIM)) * BLOCK_DIM, 1, 1}

#define BLOCK_DIM 256
#define COARSE_FACTOR 4

KERNEL
reduction_coarsened(float* output, const float* input, unsigned int n) {
  SHARED float input_s[BLOCK_DIM];

  const unsigned int segment = COARSE_FACTOR * 2 * BLOCK_DIM * group_id_x();
  const unsigned int t = local_id_x();
  const unsigned int i = segment + t;

  float sum = 0.0f;
  for (unsigned int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
    const unsigned int idx = i + tile * BLOCK_DIM;
    if (idx < n) sum += input[idx];
  }
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
