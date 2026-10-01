#include "common.h"

// Privatized text histograms. Each workgroup accumulates into its own private
// copy of the histogram so that atomics from different workgroups no longer
// contend, then merges the private copy into the global histogram.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      {length, 1, 1}

#define NUM_BINS 7

// Private copies in global memory. histo must hold num_groups * NUM_BINS
// zeroed entries; on return the final histogram is in histo[0..NUM_BINS).
KERNEL
histogram_private_global(unsigned int* histo, const char* data,
                         unsigned int length) {
  const unsigned int i = global_id_x();
  if (i < length) {
    const int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      atomic_add(&histo[group_id_x() * NUM_BINS + alphabet_position / 4], 1u);
    }
  }

  if (group_id_x() > 0) {
    // Make sure the whole workgroup has finished updating its copy.
    // Workgroup-scope fences are enough here since the copy is only written
    // by this workgroup.
    barrier();
    for (unsigned int bin = local_id_x(); bin < NUM_BINS;
         bin += group_size_x()) {
      const unsigned int value = histo[group_id_x() * NUM_BINS + bin];
      if (value > 0) {
        atomic_add(&histo[bin], value);
      }
    }
  }
}

// Private copies in LDS: much lower latency atomics, and the global
// histogram only needs NUM_BINS zeroed entries.
KERNEL
histogram_private_shared(unsigned int* histo, const char* data,
                         unsigned int length) {
  SHARED unsigned int histo_s[NUM_BINS];
  for (unsigned int bin = local_id_x(); bin < NUM_BINS; bin += group_size_x()) {
    histo_s[bin] = 0u;
  }
  barrier();

  const unsigned int i = global_id_x();
  if (i < length) {
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
