#include "common.h"
#include "merge.h"

// Basic parallel merge of sorted A (m elements) and sorted B (n elements)
// into C. Each work-item owns a contiguous range of the output, uses co-rank
// to find which parts of A and B produce it, and merges them sequentially.
// All accesses go to global memory and are uncoalesced.
//
// Workgroup: e.g. {128, 1, 1}
// Grid:      any; e.g. {ceil((m + n) / 64), 1, 1}

KERNEL
merge_basic(int* C, const int* A, int m, const int* B, int n) {
  const int tid = global_id_x();
  const int total_threads = global_size_x();
  const int elements_per_thread = (m + n + total_threads - 1) / total_threads;

  const int k_curr = min_int(tid * elements_per_thread, m + n);
  const int k_next = min_int((tid + 1) * elements_per_thread, m + n);
  if (k_curr >= k_next) return;

  const int i_curr = co_rank(k_curr, A, m, B, n);
  const int i_next = co_rank(k_next, A, m, B, n);
  const int j_curr = k_curr - i_curr;
  const int j_next = k_next - i_next;

  merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr,
                   &C[k_curr]);
}
