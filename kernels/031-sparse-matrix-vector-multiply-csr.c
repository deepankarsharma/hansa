#include "common.h"

// Sparse matrix-vector multiply y = A * x with A in CSR format. row_ptrs has
// num_rows + 1 entries; the nonzeros of row r are
// [row_ptrs[r], row_ptrs[r + 1]). One work-item per row: no atomics, but
// neighbouring work-items read memory far apart and rows of very different
// lengths cause divergence.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_rows, 1, 1}

KERNEL
spmv_csr(float* y, const unsigned int* row_ptrs, const unsigned int* col_idx,
         const float* value, unsigned int num_rows, const float* x) {
  const unsigned int row = global_id_x();
  if (row < num_rows) {
    float sum = 0.0f;
    for (unsigned int i = row_ptrs[row]; i < row_ptrs[row + 1]; i++) {
      sum += x[col_idx[i]] * value[i];
    }
    y[row] = sum;
  }
}
