#include "common.h"

// Breadth-first search, one launch per BFS level. level[v] is the BFS level of
// vertex v (UINT_MAX = not visited yet; level[root] = 0 initially). Before each
// launch the host clears *new_vertex_visited; after it, if the flag is still 0
// the search is complete, otherwise it launches again with curr_level + 1
// (starting from curr_level = 1).
//
// Three formulations that visit the same vertices:
//   - bfs_vertex_push: one work-item per vertex in the previous level, which
//     pushes to its outgoing neighbours (graph in CSR).
//   - bfs_vertex_pull: one work-item per unvisited vertex, which looks for a
//     neighbour in the previous level (graph in CSC). Good for late levels
//     where most vertices have been visited.
//   - bfs_edge: one work-item per edge (graph in COO). Exposes more
//     parallelism and balances high-degree vertices.
//
// Workgroup: e.g. {256, 1, 1}
// Grid:      {num_vertices, 1, 1} (or {num_edges, 1, 1} for bfs_edge)

#define UNVISITED 0xFFFFFFFFu

KERNEL
bfs_vertex_push(unsigned int* level, unsigned int* new_vertex_visited,
                const unsigned int* src_ptrs, const unsigned int* dst,
                unsigned int num_vertices, unsigned int curr_level) {
  const unsigned int vertex = global_id_x();
  if (vertex < num_vertices && level[vertex] == curr_level - 1) {
    for (unsigned int edge = src_ptrs[vertex]; edge < src_ptrs[vertex + 1];
         edge++) {
      const unsigned int neighbor = dst[edge];
      if (level[neighbor] == UNVISITED) {
        // Benign race: every writer stores the same value.
        level[neighbor] = curr_level;
        *new_vertex_visited = 1;
      }
    }
  }
}

KERNEL
bfs_vertex_pull(unsigned int* level, unsigned int* new_vertex_visited,
                const unsigned int* dst_ptrs, const unsigned int* src,
                unsigned int num_vertices, unsigned int curr_level) {
  const unsigned int vertex = global_id_x();
  if (vertex < num_vertices && level[vertex] == UNVISITED) {
    for (unsigned int edge = dst_ptrs[vertex]; edge < dst_ptrs[vertex + 1];
         edge++) {
      const unsigned int neighbor = src[edge];
      if (level[neighbor] == curr_level - 1) {
        level[vertex] = curr_level;
        *new_vertex_visited = 1;
        break;
      }
    }
  }
}

KERNEL
bfs_edge(unsigned int* level, unsigned int* new_vertex_visited,
         const unsigned int* src, const unsigned int* dst,
         unsigned int num_edges, unsigned int curr_level) {
  const unsigned int edge = global_id_x();
  if (edge < num_edges) {
    const unsigned int vertex = src[edge];
    if (level[vertex] == curr_level - 1) {
      const unsigned int neighbor = dst[edge];
      if (level[neighbor] == UNVISITED) {
        level[neighbor] = curr_level;
        *new_vertex_visited = 1;
      }
    }
  }
}
