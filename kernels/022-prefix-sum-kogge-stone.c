#include "common.h"

// Kogge-Stone inclusive scan (prefix sum) of each SECTION_SIZE section of the
// input. In step s every element adds the value 2^s positions to its left:
// log2(n) steps but O(n log n) additions, so it is fast for short sections
// but not work efficient.
//
// Each section is scanned independently; see 024 for a full scan of arbitrary
// length inputs.
// Workgroup: {SECTION_SIZE, 1, 1}
// Grid:      {ceil(n / SECTION_SIZE) * SECTION_SIZE, 1, 1}

#define SECTION_SIZE 256

KERNEL
prefix_sum_kogge_stone(float* Y, const float* X, unsigned int n) {
  SHARED float XY[SECTION_SIZE];

  const unsigned int t = local_id_x();
  const unsigned int i = global_id_x();
  XY[t] = i < n ? X[i] : 0.0f;

  for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2) {
    // Read and write in separate phases so that no work-item overwrites a
    // value another work-item still has to read in this step.
    barrier();
    float temp = 0.0f;
    if (t >= stride) temp = XY[t] + XY[t - stride];
    barrier();
    if (t >= stride) XY[t] = temp;
  }

  if (i < n) {
    Y[i] = XY[t];
  }
}
