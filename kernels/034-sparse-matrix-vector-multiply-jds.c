#include "common.h"

// Sparse matrix-vector multiply y = A * x with A in JDS (jagged diagonal
// storage) format. Rows are sorted by decreasing length (row_perm maps sorted
// position -> original row) and stored column-major without padding: iteration
// t holds the t-th nonzero of every row that has one, at
// [iter_ptrs[t], iter_ptrs[t + 1]). Neighbouring work-items have similar row
// lengths (little divergence) and coalesced accesses, with no padding.
//
// iter_ptrs has num_iterations + 1 entries (num_iterations = longest row).
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_rows, 1, 1}

KERNEL
spmv_jds(float* y, const unsigned int* iter_ptrs, const unsigned int* col_idx,
         const float* value, const unsigned int* row_perm,
         unsigned int num_rows, unsigned int num_iterations, const float* x) {
  const unsigned int row = global_id_x();
  if (row < num_rows) {
    float sum = 0.0f;
    for (unsigned int t = 0; t < num_iterations; t++) {
      const unsigned int iter_start = iter_ptrs[t];
      // Rows are sorted by length, so once this row has no nonzero in
      // iteration t it has none in any later one either.
      if (row >= iter_ptrs[t + 1] - iter_start) break;
      const unsigned int i = iter_start + row;
      sum += x[col_idx[i]] * value[i];
    }
    y[row_perm[row]] = sum;
  }
}
