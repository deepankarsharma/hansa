#include "common.h"
#include "merge.h"

// Tiled parallel merge. Each workgroup owns a contiguous range of C and finds
// its input ranges with co-rank. It then repeatedly loads the next TILE_SIZE
// elements of both of its A and B ranges into LDS (coalesced), and the
// work-items merge TILE_SIZE output elements from LDS. Whatever was loaded but
// not consumed is reloaded in the next iteration (see 027 for a version that
// avoids this).
//
// Workgroup: {BLOCK_DIM, 1, 1}
// Grid:      any multiple of BLOCK_DIM, e.g. {num_groups * BLOCK_DIM, 1, 1}

#define BLOCK_DIM 128
#define TILE_SIZE 1024

KERNEL
merge_tiled(int* C, const int* A, int m, const int* B, int n) {
  SHARED int A_s[TILE_SIZE];
  SHARED int B_s[TILE_SIZE];
  SHARED int block_bounds[2];

  const int t = local_id_x();

  // Output range of this workgroup and the matching input ranges.
  const int c_per_block = (m + n + num_groups_x() - 1) / num_groups_x();
  const int C_curr = min_int(group_id_x() * c_per_block, m + n);
  const int C_next = min_int((group_id_x() + 1) * c_per_block, m + n);
  if (t == 0) {
    block_bounds[0] = co_rank(C_curr, A, m, B, n);
    block_bounds[1] = co_rank(C_next, A, m, B, n);
  }
  barrier();
  const int A_curr = block_bounds[0];
  const int A_next = block_bounds[1];
  const int B_curr = C_curr - A_curr;
  const int B_next = C_next - A_next;

  const int C_length = C_next - C_curr;
  const int A_length = A_next - A_curr;
  const int B_length = B_next - B_curr;
  const int total_iterations = (C_length + TILE_SIZE - 1) / TILE_SIZE;

  int C_completed = 0;
  int A_consumed = 0;
  int B_consumed = 0;
  for (int iter = 0; iter < total_iterations; iter++) {
    const int A_tile = min_int(TILE_SIZE, A_length - A_consumed);
    const int B_tile = min_int(TILE_SIZE, B_length - B_consumed);
    const int C_tile = min_int(TILE_SIZE, C_length - C_completed);

    for (int i = t; i < A_tile; i += BLOCK_DIM) {
      A_s[i] = A[A_curr + A_consumed + i];
    }
    for (int i = t; i < B_tile; i += BLOCK_DIM) {
      B_s[i] = B[B_curr + B_consumed + i];
    }
    barrier();

    // Each work-item merges its share of this tile's output.
    const int per_thread = TILE_SIZE / BLOCK_DIM;
    const int c_curr = min_int(t * per_thread, C_tile);
    const int c_next = min_int((t + 1) * per_thread, C_tile);
    const int a_curr =
        co_rank(c_curr, (const int*)A_s, A_tile, (const int*)B_s, B_tile);
    const int a_next =
        co_rank(c_next, (const int*)A_s, A_tile, (const int*)B_s, B_tile);
    const int b_curr = c_curr - a_curr;
    const int b_next = c_next - a_next;
    merge_sequential((const int*)A_s + a_curr, a_next - a_curr,
                     (const int*)B_s + b_curr, b_next - b_curr,
                     C + C_curr + C_completed + c_curr);

    // How much of each tile did this iteration use up?
    A_consumed +=
        co_rank(C_tile, (const int*)A_s, A_tile, (const int*)B_s, B_tile);
    C_completed += C_tile;
    B_consumed = C_completed - A_consumed;
    barrier();
  }
}
