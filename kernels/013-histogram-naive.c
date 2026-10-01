#include "common.h"

// Naive text histogram: letters a-z are grouped into bins of four
// (a-d, e-h, ..., y-z) and every work-item increments the global bin for its
// character with an atomic add. Heavy contention on only NUM_BINS addresses.
//
// histo must have NUM_BINS entries and be zeroed before launch.
// Workgroup: e.g. {256, 1, 1}
// Grid:      {length, 1, 1}

#define NUM_BINS 7

KERNEL
histogram_naive(unsigned int* histo, const char* data, unsigned int length) {
  const unsigned int i = global_id_x();
  if (i < length) {
    const int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) {
      atomic_add(&histo[alphabet_position / 4], 1u);
    }
  }
}
