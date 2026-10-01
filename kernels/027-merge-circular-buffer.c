#include "common.h"
#include "merge.h"

// Tiled parallel merge with circular LDS buffers. Same structure as 026, but
// unconsumed elements stay in LDS: each iteration only loads as many new
// elements as were consumed in the previous one, into the slots they freed.
// This removes the redundant global memory reloads of the plain tiled merge.
//
// Workgroup: {BLOCK_DIM, 1, 1}
// Grid:      any multiple of BLOCK_DIM, e.g. {num_groups * BLOCK_DIM, 1, 1}

#define BLOCK_DIM 128
#define TILE_SIZE 1024

// co_rank over two circular buffers whose logical element 0 lives at
// A_start / B_start.
static int
co_rank_circular(int k, const LOCAL int* A, int m, const LOCAL int* B, int n,
                 int A_start, int B_start) {
  int i = k < m ? k : m;
  int j = k - i;
  int i_low = k - n > 0 ? k - n : 0;
  int j_low = k - m > 0 ? k - m : 0;
  while (1) {
    const int i_cir = (A_start + i) % TILE_SIZE;
    const int i_m_1_cir = (A_start + i - 1) % TILE_SIZE;
    const int j_cir = (B_start + j) % TILE_SIZE;
    const int j_m_1_cir = (B_start + j - 1) % TILE_SIZE;
    if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      j += delta;
      i -= delta;
    } else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      return i;
    }
  }
}

// merge_sequential over two circular buffers, starting at logical positions
// A_start / B_start.
static void
merge_sequential_circular(const LOCAL int* A, int m, const LOCAL int* B, int n,
                          int* C, int A_start, int B_start) {
  int i = 0, j = 0, k = 0;
  while (i < m && j < n) {
    const int a = A[(A_start + i) % TILE_SIZE];
    const int b = B[(B_start + j) % TILE_SIZE];
    if (a <= b) {
      C[k++] = a;
      i++;
    } else {
      C[k++] = b;
      j++;
    }
  }
  while (i < m) C[k++] = A[(A_start + i++) % TILE_SIZE];
  while (j < n) C[k++] = B[(B_start + j++) % TILE_SIZE];
}

KERNEL
merge_circular_buffer(int* C, const int* A, int m, const int* B, int n) {
  SHARED int A_s[TILE_SIZE];
  SHARED int B_s[TILE_SIZE];
  SHARED int block_bounds[2];

  const int t = local_id_x();

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
  int A_consumed = 0;  // Consumed from global memory so far.
  int B_consumed = 0;
  int A_s_start = 0;  // Logical start of the circular buffers.
  int B_s_start = 0;
  int A_s_consumed = TILE_SIZE;  // Slots freed in the previous iteration
  int B_s_consumed = TILE_SIZE;  // (everything, initially).

  for (int iter = 0; iter < total_iterations; iter++) {
    // The buffer holds TILE_SIZE - A_s_consumed still-unconsumed elements
    // (fewer near the end of the input); refill the freed slots behind them.
    const int A_kept = TILE_SIZE - A_s_consumed;
    const int B_kept = TILE_SIZE - B_s_consumed;
    for (int i = t; i < A_s_consumed; i += BLOCK_DIM) {
      if (A_kept + i < A_length - A_consumed) {
        A_s[(A_s_start + A_kept + i) % TILE_SIZE] =
            A[A_curr + A_consumed + A_kept + i];
      }
    }
    for (int i = t; i < B_s_consumed; i += BLOCK_DIM) {
      if (B_kept + i < B_length - B_consumed) {
        B_s[(B_s_start + B_kept + i) % TILE_SIZE] =
            B[B_curr + B_consumed + B_kept + i];
      }
    }
    barrier();

    const int A_tile = min_int(TILE_SIZE, A_length - A_consumed);
    const int B_tile = min_int(TILE_SIZE, B_length - B_consumed);
    const int C_tile = min_int(TILE_SIZE, C_length - C_completed);

    const int per_thread = TILE_SIZE / BLOCK_DIM;
    const int c_curr = min_int(t * per_thread, C_tile);
    const int c_next = min_int((t + 1) * per_thread, C_tile);
    const int a_curr = co_rank_circular(c_curr, A_s, A_tile, B_s, B_tile,
                                        A_s_start, B_s_start);
    const int a_next = co_rank_circular(c_next, A_s, A_tile, B_s, B_tile,
                                        A_s_start, B_s_start);
    const int b_curr = c_curr - a_curr;
    const int b_next = c_next - a_next;
    merge_sequential_circular(A_s, a_next - a_curr, B_s, b_next - b_curr,
                              C + C_curr + C_completed + c_curr,
                              A_s_start + a_curr, B_s_start + b_curr);

    A_s_consumed = co_rank_circular(C_tile, A_s, A_tile, B_s, B_tile, A_s_start,
                                    B_s_start);
    B_s_consumed = C_tile - A_s_consumed;
    A_consumed += A_s_consumed;
    B_consumed += B_s_consumed;
    C_completed += C_tile;
    A_s_start = (A_s_start + A_s_consumed) % TILE_SIZE;
    B_s_start = (B_s_start + B_s_consumed) % TILE_SIZE;
    barrier();
  }
}
