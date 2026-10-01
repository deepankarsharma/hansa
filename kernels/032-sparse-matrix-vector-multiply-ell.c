#include "common.h"

// Sparse matrix-vector multiply y = A * x with A in ELL format: every row is
// padded to the same length and the padded matrix is stored column-major, so
// nonzero t of row r is at index t * num_rows + r. When work-items (one per
// row) step through their t-th nonzero together, their accesses are
// consecutive and coalesce. nnz_per_row lets the loop skip the padding.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_rows, 1, 1}

KERNEL
spmv_ell(float* y, const unsigned int* nnz_per_row, const unsigned int* col_idx,
         const float* value, unsigned int num_rows, const float* x) {
  const unsigned int row = global_id_x();
  if (row < num_rows) {
    float sum = 0.0f;
    for (unsigned int t = 0; t < nnz_per_row[row]; t++) {
      const unsigned int i = t * num_rows + row;
      sum += x[col_idx[i]] * value[i];
    }
    y[row] = sum;
  }
}
