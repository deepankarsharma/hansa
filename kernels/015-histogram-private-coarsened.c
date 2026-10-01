#include "common.h"

// Privatized (LDS) text histogram with thread coarsening: launch fewer
// work-items than characters and let each one process several characters,
// which amortizes the cost of initializing and merging the private copy.
//
// histo must have NUM_BINS entries and be zeroed before launch.
// Workgroup: e.g. {256, 1, 1}
// Grid:      e.g. {ceil(length / CFACTOR), 1, 1}

#define NUM_BINS 7
#define CFACTOR 32

// Contiguous partitioning: work-item t handles characters
// [t * CFACTOR, (t + 1) * CFACTOR). Simple, but neighbouring work-items access
// memory CFACTOR elements apart, so the accesses are not coalesced.
KERNEL
histogram_private_coarsened_contiguous(unsigned int* histo, const char* data,
                                       unsigned int length) {
  SHARED unsigned int histo_s[NUM_BINS];
  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    histo_s[bin] = 0u;
  }
  barrier();

  const unsigned int tid = global_id_x();
  for (unsigned int i = tid * CFACTOR; i < (tid + 1) * CFACTOR && i < length;
       i++) {
    const int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      atomic_add_local(&histo_s[alphabet_position / 4], 1u);
    }
  }
  barrier();

  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    const unsigned int value = histo_s[bin];
    if (value > 0) {
      atomic_add(&histo[bin], value);
    }
  }
}

// Interleaved partitioning: grid-stride loop, so in every iteration
// consecutive work-items read consecutive characters (coalesced). Works with
// any grid size.
KERNEL
histogram_private_coarsened_interleaved(unsigned int* histo, const char* data,
                                        unsigned int length) {
  SHARED unsigned int histo_s[NUM_BINS];
  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    histo_s[bin] = 0u;
  }
  barrier();

  const unsigned int stride = global_size_x();
  for (unsigned int i = global_id_x(); i < length; i += stride) {
    const int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      atomic_add_local(&histo_s[alphabet_position / 4], 1u);
    }
  }
  barrier();

  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    const unsigned int value = histo_s[bin];
    if (value > 0) {
      atomic_add(&histo[bin], value);
    }
  }
}
