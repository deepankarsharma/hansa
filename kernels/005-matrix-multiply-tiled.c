#include <stdint.h>

#define TILE_SIZE 16  // Adjust based on GPU's shared memory limits

__attribute__((visibility("default"), amdgpu_kernel)) 
void matrix_multiply_tiled(
    float* C, const float* A, const float* B, int N, int M, int K) {

    // Thread and block indices
    int row = __builtin_amdgcn_workgroup_id_y() * TILE_SIZE +
              __builtin_amdgcn_workitem_id_y();
    int col = __builtin_amdgcn_workgroup_id_x() * TILE_SIZE +
              __builtin_amdgcn_workitem_id_x();

    // Accumulator for the result
    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (M + TILE_SIZE - 1) / TILE_SIZE; tile++) {
      // __builtin_amdgcn_scratch_alloc
        // Load tile elements into shared memory (manually managed)
      static __attribute__((address_space(3))) float tile_A[TILE_SIZE][TILE_SIZE];
      static __attribute__((address_space(3))) float tile_B[TILE_SIZE][TILE_SIZE];

      int tiled_row = row;
      int tiled_col = col;

      // Load A tile from global memory
      if (tiled_row < N && tile * TILE_SIZE + __builtin_amdgcn_workitem_id_x() < M) {
        tile_A[__builtin_amdgcn_workitem_id_y()][__builtin_amdgcn_workitem_id_x()] =
          A[tiled_row * M + (tile * TILE_SIZE + __builtin_amdgcn_workitem_id_x())];
      } else {
        tile_A[__builtin_amdgcn_workitem_id_y()][__builtin_amdgcn_workitem_id_x()] = 0.0f;
      }

        // Load B tile from global memory
        if (tiled_col < K && tile * TILE_SIZE + __builtin_amdgcn_workitem_id_y() < M) {
            tile_B[__builtin_amdgcn_workitem_id_y()][__builtin_amdgcn_workitem_id_x()] =
                B[(tile * TILE_SIZE + __builtin_amdgcn_workitem_id_y()) * K + tiled_col];
        } else {
            tile_B[__builtin_amdgcn_workitem_id_y()][__builtin_amdgcn_workitem_id_x()] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded data
        __builtin_amdgcn_s_barrier();

        // Perform computation on the loaded tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[__builtin_amdgcn_workitem_id_y()][i] *
                   tile_B[i][__builtin_amdgcn_workitem_id_x()];
        }

        // Synchronize before loading the next tile
        __builtin_amdgcn_s_barrier();
    }

    // Store the result in the output matrix C
    if (row < N && col < K) {
        C[row * K + col] = sum;
    }
}

