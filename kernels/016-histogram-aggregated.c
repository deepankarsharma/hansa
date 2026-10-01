#include "common.h"

// Privatized, coarsened (interleaved) text histogram with aggregation: each
// work-item tracks a run of identical consecutive bins in a register and only
// issues an atomic when the bin changes. Big win for inputs with long runs
// of the same value (e.g. images with large uniform regions).
//
// histo must have NUM_BINS entries and be zeroed before launch.
// Workgroup: e.g. {256, 1, 1}
// Grid:      any, e.g. {ceil(length / 32), 1, 1}

#define NUM_BINS 7

KERNEL
histogram_aggregated(unsigned int* histo, const char* data,
                     unsigned int length) {
  SHARED unsigned int histo_s[NUM_BINS];
  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    histo_s[bin] = 0u;
  }
  barrier();

  unsigned int accumulator = 0;
  int prev_bin_idx = -1;
  const unsigned int stride = global_size_x();
  for (unsigned int i = global_id_x(); i < length; i += stride) {
    const int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      const int bin = alphabet_position / 4;
      if (bin == prev_bin_idx) {
        accumulator++;
      } else {
        if (accumulator > 0) {
          atomic_add_local(&histo_s[prev_bin_idx], accumulator);
        }
        accumulator = 1;
        prev_bin_idx = bin;
      }
    }
  }
  if (accumulator > 0) {
    atomic_add_local(&histo_s[prev_bin_idx], accumulator);
  }
  barrier();

  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    const unsigned int value = histo_s[bin];
    if (value > 0) {
      atomic_add(&histo[bin], value);
    }
  }
}
