#include "common.h"

// Frontier-based BFS (graph in CSR), one launch per BFS level. Instead of
// launching a work-item for every vertex and having most of them find nothing
// to do, only the vertices in the previous frontier are processed, and newly
// visited vertices are appended to the current frontier.
//
// level[v] = UINT_MAX for unvisited vertices, level[root] = 0. The host
// starts with prev_frontier = {root}, curr_level = 1, and for each level
// zeroes *num_curr_frontier, launches, then swaps the frontiers; it stops
// when the new frontier is empty.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_prev_frontier, 1, 1}

#define UNVISITED 0xFFFFFFFFu

KERNEL
bfs_frontier(unsigned int* level, unsigned int* curr_frontier,
             unsigned int* num_curr_frontier, const unsigned int* prev_frontier,
             unsigned int num_prev_frontier, const unsigned int* src_ptrs,
             const unsigned int* dst, unsigned int curr_level) {
  const unsigned int i = global_id_x();
  if (i < num_prev_frontier) {
    const unsigned int vertex = prev_frontier[i];
    for (unsigned int edge = src_ptrs[vertex]; edge < src_ptrs[vertex + 1];
         edge++) {
      const unsigned int neighbor = dst[edge];
      // The compare-and-swap makes sure each vertex is claimed (and added to
      // the frontier) exactly once.
      if (atomic_cas(&level[neighbor], UNVISITED, curr_level) == UNVISITED) {
        const unsigned int idx = atomic_add(num_curr_frontier, 1u);
        curr_frontier[idx] = neighbor;
      }
    }
  }
}
