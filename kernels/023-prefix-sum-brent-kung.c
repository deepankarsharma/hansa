#include "common.h"

// Brent-Kung inclusive scan of each SECTION_SIZE section of the input. A
// reduction tree phase builds partial sums at power-of-two positions, then a
// reverse tree phase distributes them: 2 log2(n) steps but only O(n)
// additions (work efficient). Each work-item handles two elements, so a
// workgroup of BLOCK_DIM work-items scans 2 * BLOCK_DIM elements.
//
// Workgroup: {BLOCK_DIM, 1, 1}
// Grid:      {ceil(n / SECTION_SIZE) * BLOCK_DIM, 1, 1}

#define BLOCK_DIM 256
#define SECTION_SIZE (2 * BLOCK_DIM)

KERNEL
prefix_sum_brent_kung(float* Y, const float* X, unsigned int n) {
  SHARED float XY[SECTION_SIZE];

  const unsigned int t = local_id_x();
  const unsigned int i = group_id_x() * SECTION_SIZE + t;
  XY[t] = i < n ? X[i] : 0.0f;
  XY[t + BLOCK_DIM] = i + BLOCK_DIM < n ? X[i + BLOCK_DIM] : 0.0f;

  // Reduction tree phase.
  for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
    barrier();
    const unsigned int index = (t + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) {
      XY[index] += XY[index - stride];
    }
  }

  // Reverse tree (distribution) phase.
  for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
    barrier();
    const unsigned int index = (t + 1) * 2 * stride - 1;
    if (index + stride < SECTION_SIZE) {
      XY[index + stride] += XY[index];
    }
  }
  barrier();

  if (i < n) Y[i] = XY[t];
  if (i + BLOCK_DIM < n) Y[i + BLOCK_DIM] = XY[t + BLOCK_DIM];
}
