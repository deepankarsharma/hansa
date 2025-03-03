#include <stdint.h>

__attribute__((visibility("default"), amdgpu_kernel))
void matrix_multiply(
    float* C, const float* A, const float* B, int N, int M, int K) {
    
    int row = __builtin_amdgcn_workgroup_id_y() * __builtin_amdgcn_workgroup_size_y() +
              __builtin_amdgcn_workitem_id_y();
    int col = __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() +
              __builtin_amdgcn_workitem_id_x();

    if (row >= N || col >= K) return; // Boundary check

    float sum = 0.0f;
    
    // Compute dot product of row in A and column in B
    for (int i = 0; i < M; i++) {
        sum += A[row * M + i] * B[i * K + col];
    }

    // Store result in C
    C[row * K + col] = sum;
}
