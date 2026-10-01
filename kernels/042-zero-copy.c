#include "common.h"

// Zero-copy access to host memory. The kernel itself is an ordinary SAXPY;
// what makes it "zero-copy" is how the host allocates x and y: in
// fine-grained system memory (e.g. Engine's system_region_, or memory from
// hsa_amd_memory_pool_allocate on the CPU agent's pool / hsa_amd_memory_lock),
// so the GPU reads and writes host memory directly over the bus with no
// explicit copies. This pays off when the data is touched only once, or on
// APUs (such as gfx1103) where CPU and GPU share the same physical memory.
//
// Every element is touched exactly once with fully coalesced accesses, which
// is the access pattern that suits zero-copy memory. The grid-stride loop
// lets any grid size cover the whole array.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      any, e.g. {min(n, 65536), 1, 1}

KERNEL
zero_copy_saxpy(float* y, const float* x, float a, unsigned int n) {
  const unsigned int stride = global_size_x();
  for (unsigned int i = global_id_x(); i < n; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}
