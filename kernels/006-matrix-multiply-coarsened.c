#include "common.h"

// Tiled matrix multiply with thread coarsening: each work-item computes
// COARSE_FACTOR output elements in the same row, spaced TILE_SIZE columns
// apart, so every tile of A loaded into LDS is reused COARSE_FACTOR times.
//
// C (N x K) = A (N x M) * B (M x K)
// Workgroup: {TILE_SIZE, TILE_SIZE, 1}
// Grid:      {ceil(K / (TILE_SIZE * COARSE_FACTOR)) * TILE_SIZE,
//             ceil(N / TILE_SIZE) * TILE_SIZE, 1}

#define TILE_SIZE 16
#define COARSE_FACTOR 4

KERNEL
matrix_multiply_coarsened(float* C, const float* A, const float* B, int N,
                          int M, int K) {
  SHARED float tile_A[TILE_SIZE][TILE_SIZE];
  SHARED float tile_B[TILE_SIZE][TILE_SIZE];

  const int tx = local_id_x();
  const int ty = local_id_y();

  const int row = group_id_y() * TILE_SIZE + ty;
  const int col_start = group_id_x() * TILE_SIZE * COARSE_FACTOR + tx;

  float sum[COARSE_FACTOR];
  for (int c = 0; c < COARSE_FACTOR; c++) sum[c] = 0.0f;

  for (int tile = 0; tile < (M + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    // The A tile is loaded once per phase...
    const int a_col = tile * TILE_SIZE + tx;
    tile_A[ty][tx] = (row < N && a_col < M) ? A[row * M + a_col] : 0.0f;

    // ...and reused against COARSE_FACTOR different B tiles.
    for (int c = 0; c < COARSE_FACTOR; c++) {
      const int col = col_start + c * TILE_SIZE;
      const int b_row = tile * TILE_SIZE + ty;
      tile_B[ty][tx] = (b_row < M && col < K) ? B[b_row * K + col] : 0.0f;
      barrier();

      for (int i = 0; i < TILE_SIZE; i++) {
        sum[c] += tile_A[ty][i] * tile_B[i][tx];
      }
      barrier();
    }
  }

  for (int c = 0; c < COARSE_FACTOR; c++) {
    const int col = col_start + c * TILE_SIZE;
    if (row < N && col < K) {
      C[row * K + col] = sum[c];
    }
  }
}
