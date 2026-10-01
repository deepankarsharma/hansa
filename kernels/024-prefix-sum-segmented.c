#include "common.h"

// Hierarchical (segmented) inclusive scan for arbitrary length inputs, in
// three launches:
//
//   1. prefix_sum_segmented_scan_blocks(Y, X, S, n)
//        Brent-Kung scan of every SECTION_SIZE segment of X into Y; the total
//        of segment b is written to S[b].
//        Grid: {num_segments * BLOCK_DIM, 1, 1}, num_segments =
//        ceil(n / SECTION_SIZE).
//   2. prefix_sum_segmented_scan_blocks(S, S, NULL, num_segments)
//        Scan of the segment totals in place (one workgroup suffices as long
//        as num_segments <= SECTION_SIZE, i.e. n <= SECTION_SIZE^2 = 1M
//        elements; for more, recurse).
//        Grid: {BLOCK_DIM, 1, 1}
//   3. prefix_sum_segmented_add_sums(Y, S, n)
//        Adds the scanned total of all preceding segments to every element.
//        Grid: same as step 1.
//
// Workgroup: {BLOCK_DIM, 1, 1} for all three.

#define BLOCK_DIM 512
#define SECTION_SIZE (2 * BLOCK_DIM)

KERNEL
prefix_sum_segmented_scan_blocks(float* Y, const float* X, float* S,
                                 unsigned int n) {
  SHARED float XY[SECTION_SIZE];

  const unsigned int t = local_id_x();
  const unsigned int i = group_id_x() * SECTION_SIZE + t;
  XY[t] = i < n ? X[i] : 0.0f;
  XY[t + BLOCK_DIM] = i + BLOCK_DIM < n ? X[i + BLOCK_DIM] : 0.0f;

  for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
    barrier();
    const unsigned int index = (t + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) {
      XY[index] += XY[index - stride];
    }
  }
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

  if (S && t == BLOCK_DIM - 1) {
    S[group_id_x()] = XY[SECTION_SIZE - 1];
  }
}

KERNEL
prefix_sum_segmented_add_sums(float* Y, const float* S, unsigned int n) {
  const unsigned int segment = group_id_x();
  if (segment == 0) return;

  const float preceding = S[segment - 1];
  const unsigned int i = segment * SECTION_SIZE + local_id_x();
  if (i < n) Y[i] += preceding;
  if (i + BLOCK_DIM < n) Y[i + BLOCK_DIM] += preceding;
}
