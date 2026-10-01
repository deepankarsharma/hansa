#ifndef HANSA_KERNELS_MERGE_H
#define HANSA_KERNELS_MERGE_H

// Helpers shared by the merge (025 - 027) and merge sort (029) kernels.

// Sequentially merges sorted A[0..m) and B[0..n) into C[0..m+n). Stable:
// on ties, elements of A come first.
static inline void
merge_sequential(const int* A, int m, const int* B, int n, int* C) {
  int i = 0, j = 0, k = 0;
  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }
  while (i < m) C[k++] = A[i++];
  while (j < n) C[k++] = B[j++];
}

// Co-rank: for output position k of merge(A, B), returns i such that the
// first k outputs are made of A[0..i) and B[0..k-i). Binary search, so
// O(log(min(m, n))).
static inline int
co_rank(int k, const int* A, int m, const int* B, int n) {
  int i = k < m ? k : m;
  int j = k - i;
  int i_low = k - n > 0 ? k - n : 0;
  int j_low = k - m > 0 ? k - m : 0;
  while (1) {
    if (i > 0 && j < n && A[i - 1] > B[j]) {
      // i is too high.
      const int delta = (i - i_low + 1) >> 1;
      j_low = j;
      j += delta;
      i -= delta;
    } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
      // j is too high.
      const int delta = (j - j_low + 1) >> 1;
      i_low = i;
      i += delta;
      j -= delta;
    } else {
      return i;
    }
  }
}

#endif  // HANSA_KERNELS_MERGE_H
