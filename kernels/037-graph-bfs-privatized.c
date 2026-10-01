#include "common.h"

// Frontier-based BFS with a privatized frontier. Like 036, but each workgroup
// first appends to a small frontier in LDS (cheap workgroup-scope atomics),
// then reserves space in the global frontier with a single atomic and copies
// its local frontier over with coalesced writes. If the local frontier fills
// up, the overflow goes straight to the global frontier.
//
// Same host protocol as 036.
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_prev_frontier, 1, 1}

#define UNVISITED 0xFFFFFFFFu
#define LOCAL_FRONTIER_CAPACITY 1024

KERNEL
bfs_frontier_privatized(unsigned int* level, unsigned int* curr_frontier,
                        unsigned int* num_curr_frontier,
                        const unsigned int* prev_frontier,
                        unsigned int num_prev_frontier,
                        const unsigned int* src_ptrs, const unsigned int* dst,
                        unsigned int curr_level) {
  SHARED unsigned int curr_frontier_s[LOCAL_FRONTIER_CAPACITY];
  SHARED unsigned int num_curr_frontier_s;
  SHARED unsigned int curr_frontier_start_s;

  if (local_id_x() == 0) {
    num_curr_frontier_s = 0;
  }
  barrier();

  const unsigned int i = global_id_x();
  if (i < num_prev_frontier) {
    const unsigned int vertex = prev_frontier[i];
    for (unsigned int edge = src_ptrs[vertex]; edge < src_ptrs[vertex + 1];
         edge++) {
      const unsigned int neighbor = dst[edge];
      if (atomic_cas(&level[neighbor], UNVISITED, curr_level) == UNVISITED) {
        const unsigned int idx_s = atomic_add_local(&num_curr_frontier_s, 1u);
        if (idx_s < LOCAL_FRONTIER_CAPACITY) {
          curr_frontier_s[idx_s] = neighbor;
        } else {
          // Local frontier is full: undo the reservation and spill to the
          // global frontier.
          atomic_add_local(&num_curr_frontier_s, (unsigned int)-1);
          const unsigned int idx = atomic_add(num_curr_frontier, 1u);
          curr_frontier[idx] = neighbor;
        }
      }
    }
  }
  barrier();

  // Reserve space in the global frontier for the whole local frontier.
  if (local_id_x() == 0) {
    curr_frontier_start_s = atomic_add(num_curr_frontier, num_curr_frontier_s);
  }
  barrier();

  // Commit the local frontier to the global frontier.
  const unsigned int count = num_curr_frontier_s;
  for (unsigned int idx_s = local_id_x(); idx_s < count;
       idx_s += group_size_x()) {
    curr_frontier[curr_frontier_start_s + idx_s] = curr_frontier_s[idx_s];
  }
}
