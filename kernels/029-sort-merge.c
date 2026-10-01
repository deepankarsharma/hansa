#include "common.h"
#include "merge.h"

// Bottom-up parallel merge sort. The host calls merge_sort_pass with
// width = 1, 2, 4, ... while width < n, swapping in / out after every pass.
// In each pass adjacent sorted runs of `width` elements are merged pairwise
// into runs of 2 * width. Work is split by output position: every work-item
// produces ELEMENTS_PER_THREAD consecutive outputs, using co-rank to locate
// its inputs, so the load stays balanced even when there are only a few huge
// runs left.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      {ceil(n / ELEMENTS_PER_THREAD), 1, 1}

#define ELEMENTS_PER_THREAD 8

KERNEL
merge_sort_pass(int* out, const int* in, int n, int width) {
  int k = global_id_x() * ELEMENTS_PER_THREAD;
  const int k_end = min_int(k + ELEMENTS_PER_THREAD, n);

  // A work-item's output range can span several pairs of runs when the runs
  // are short, so walk pair by pair.
  while (k < k_end) {
    const int base = (k / (2 * width)) * (2 * width);
    const int* A = in + base;
    const int m = min_int(width, n - base);
    const int* B = A + m;
    const int nb = min_int(width, n - base - m);

    const int seg_end = min_int(k_end, base + m + nb);
    const int i_curr = co_rank(k - base, A, m, B, nb);
    const int i_next = co_rank(seg_end - base, A, m, B, nb);
    const int j_curr = (k - base) - i_curr;
    const int j_next = (seg_end - base) - i_next;
    merge_sequential(A + i_curr, i_next - i_curr, B + j_curr, j_next - j_curr,
                     out + k);
    k = seg_end;
  }
}
