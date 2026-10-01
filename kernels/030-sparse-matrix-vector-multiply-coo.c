#include "common.h"

// Sparse matrix-vector multiply y += A * x with A in COO format (parallel
// arrays of row index, column index and value for every nonzero). One
// work-item per nonzero; several nonzeros share a row, so the update of y
// needs an atomic.
//
// y must be zeroed (or hold the values to accumulate onto) before launch.
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_nonzeros, 1, 1}

KERNEL
spmv_coo(float* y, const unsigned int* row_idx, const unsigned int* col_idx,
         const float* value, unsigned int num_nonzeros, const float* x) {
  const unsigned int i = global_id_x();
  if (i < num_nonzeros) {
    atomic_add(&y[row_idx[i]], x[col_idx[i]] * value[i]);
  }
}
