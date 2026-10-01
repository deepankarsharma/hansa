#include "common.h"

// Parallel LSD radix sort of unsigned keys, one bit per iteration. Each
// iteration is a stable split on bit `iter`: keys with a 0 bit keep their
// relative order and go first, keys with a 1 bit follow. The destination of a
// key comes from an exclusive scan of the bits:
//
//   ones_before = number of 1 bits before position i
//   dst = bit == 0 ? i - ones_before : n - total_ones + ones_before
//
// For each bit iter = 0..31 the host launches (num_blocks = ceil(n /
// BLOCK_DIM)):
//   1. radix_sort_local_scan(local_scan, block_ones, keys_in, n, iter)
//        Grid: {num_blocks * BLOCK_DIM, 1, 1}
//   2. radix_sort_scan_block_sums(block_ones, num_blocks)
//        block_ones needs num_blocks + 1 entries. Grid: {BLOCK_DIM, 1, 1}
//   3. radix_sort_scatter(keys_out, keys_in, local_scan, block_ones, n, iter)
//        Grid: same as step 1.
// and then swaps keys_in / keys_out.
//
// Workgroup: {BLOCK_DIM, 1, 1}

#define BLOCK_DIM 256

// Inclusive Kogge-Stone scan of BLOCK_DIM values in LDS. Returns the
// inclusive prefix for work-item t.
static unsigned int
block_inclusive_scan(LOCAL unsigned int* s, unsigned int t, unsigned int v) {
  s[t] = v;
  for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2) {
    barrier();
    const unsigned int temp = t >= stride ? s[t] + s[t - stride] : s[t];
    barrier();
    s[t] = temp;
  }
  barrier();
  return s[t];
}

// Per-workgroup exclusive scan of the key bits.
KERNEL
radix_sort_local_scan(unsigned int* local_scan, unsigned int* block_ones,
                      const unsigned int* keys, unsigned int n,
                      unsigned int iter) {
  SHARED unsigned int scan_s[BLOCK_DIM];

  const unsigned int t = local_id_x();
  const unsigned int i = group_id_x() * BLOCK_DIM + t;
  const unsigned int bit = i < n ? (keys[i] >> iter) & 1u : 0u;

  const unsigned int inclusive = block_inclusive_scan(scan_s, t, bit);
  if (i < n) {
    local_scan[i] = inclusive - bit;
  }
  if (t == BLOCK_DIM - 1) {
    block_ones[group_id_x()] = inclusive;
  }
}

// Single-workgroup exclusive scan of the per-workgroup 1-bit counts, chunk by
// chunk with a running carry so any number of blocks is supported. The grand
// total is stored in block_ones[num_blocks].
KERNEL
radix_sort_scan_block_sums(unsigned int* block_ones, unsigned int num_blocks) {
  SHARED unsigned int scan_s[BLOCK_DIM];

  const unsigned int t = local_id_x();
  unsigned int carry = 0;
  for (unsigned int base = 0; base < num_blocks; base += BLOCK_DIM) {
    const unsigned int i = base + t;
    const unsigned int v = i < num_blocks ? block_ones[i] : 0u;
    const unsigned int inclusive = block_inclusive_scan(scan_s, t, v);
    if (i < num_blocks) {
      block_ones[i] = carry + inclusive - v;
    }
    carry += scan_s[BLOCK_DIM - 1];
    barrier();
  }
  if (t == 0) {
    block_ones[num_blocks] = carry;
  }
}

KERNEL
radix_sort_scatter(unsigned int* keys_out, const unsigned int* keys_in,
                   const unsigned int* local_scan,
                   const unsigned int* block_ones, unsigned int n,
                   unsigned int iter) {
  const unsigned int i = global_id_x();
  if (i >= n) return;

  const unsigned int num_blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
  const unsigned int key = keys_in[i];
  const unsigned int bit = (key >> iter) & 1u;
  const unsigned int ones_before = block_ones[group_id_x()] + local_scan[i];
  const unsigned int total_ones = block_ones[num_blocks];
  const unsigned int dst =
      bit == 0 ? i - ones_before : n - total_ones + ones_before;
  keys_out[dst] = key;
}
