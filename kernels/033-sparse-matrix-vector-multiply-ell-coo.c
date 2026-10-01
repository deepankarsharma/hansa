#include "common.h"

// Sparse matrix-vector multiply y = A * x with A in hybrid ELL-COO format. A
// few unusually long rows would force a lot of ELL padding, so the nonzeros
// beyond the first ell_width of each row are moved into a COO "overflow"
// part. The host launches:
//
//   1. spmv_hybrid_ell  - one work-item per row, writes y.
//        Grid: {num_rows, 1, 1}
//   2. spmv_hybrid_coo  - one work-item per overflow nonzero, atomically adds
//        onto the y written by step 1.
//        Grid: {num_coo_nonzeros, 1, 1}
//
// Workgroup: e.g. {256, 1, 1}

KERNEL
spmv_hybrid_ell(float* y, const unsigned int* nnz_per_row,
                const unsigned int* ell_col_idx, const float* ell_value,
                unsigned int num_rows, const float* x) {
  const unsigned int row = global_id_x();
  if (row < num_rows) {
    float sum = 0.0f;
    for (unsigned int t = 0; t < nnz_per_row[row]; t++) {
      const unsigned int i = t * num_rows + row;
      sum += x[ell_col_idx[i]] * ell_value[i];
    }
    y[row] = sum;
  }
}

KERNEL
spmv_hybrid_coo(float* y, const unsigned int* coo_row_idx,
                const unsigned int* coo_col_idx, const float* coo_value,
                unsigned int num_coo_nonzeros, const float* x) {
  const unsigned int i = global_id_x();
  if (i < num_coo_nonzeros) {
    atomic_add(&y[coo_row_idx[i]], x[coo_col_idx[i]] * coo_value[i]);
  }
}
