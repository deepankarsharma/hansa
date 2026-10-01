#include "common.h"

// Quadtree construction over a set of 2D points.
//
// The CUDA version of this example uses dynamic parallelism: each node's
// workgroup partitions its points into four quadrants and launches a child
// grid for the four children. Plain HSA has no device-side enqueue from C, so
// the tree is instead built one level per launch:
//
//   nodes[0] = root {bounds, begin = 0, end = num_points}, *node_count = 1
//   level_begin = 0, level_end = 1, in = points, out = scratch
//   for depth = 0 .. while level_begin < level_end:
//     launch quadtree_build_level with level_end - level_begin workgroups
//     level_begin = level_end, level_end = *node_count, swap(in, out)
//
// Every launch reorders all points so that each node's points are
// contiguous, writing into the other buffer; after the final launch `in`
// holds the points sorted by leaf. nodes must have room for every node that
// can be created (at most 1 + 4 + ... + 4^max_depth).
//
// Workgroup: e.g. {128, 1, 1}
// Grid:      {(level_end - level_begin) * 128, 1, 1}

typedef struct {
  float min_x, min_y, max_x, max_y;  // Bounding box.
  int begin, end;                    // Range of points in this node.
  int first_child;                   // Index of the first of 4 children,
                                     // -1 for a leaf.
  int depth;
} quadtree_node_t;

KERNEL
quadtree_build_level(quadtree_node_t* nodes, int* node_count, int level_begin,
                     float* out_x, float* out_y, const float* in_x,
                     const float* in_y, int max_depth,
                     int min_points_per_node) {
  SHARED unsigned int quadrant_count[4];
  SHARED unsigned int quadrant_offset[4];

  quadtree_node_t* node = &nodes[level_begin + group_id_x()];
  const int t = local_id_x();
  const int begin = node->begin;
  const int end = node->end;

  // Leaf: just carry the points over to the output buffer.
  if (node->depth >= max_depth || end - begin <= min_points_per_node) {
    for (int i = begin + t; i < end; i += group_size_x()) {
      out_x[i] = in_x[i];
      out_y[i] = in_y[i];
    }
    if (t == 0) node->first_child = -1;
    return;
  }

  const float mid_x = 0.5f * (node->min_x + node->max_x);
  const float mid_y = 0.5f * (node->min_y + node->max_y);

  // Quadrants: 0 = top-left, 1 = top-right, 2 = bottom-left,
  // 3 = bottom-right (y grows upwards).
  if (t < 4) quadrant_count[t] = 0;
  barrier();

  for (int i = begin + t; i < end; i += group_size_x()) {
    const int q = (in_x[i] >= mid_x) + 2 * (in_y[i] < mid_y);
    atomic_add_local(&quadrant_count[q], 1u);
  }
  barrier();

  if (t == 0) {
    quadrant_offset[0] = begin;
    quadrant_offset[1] = quadrant_offset[0] + quadrant_count[0];
    quadrant_offset[2] = quadrant_offset[1] + quadrant_count[1];
    quadrant_offset[3] = quadrant_offset[2] + quadrant_count[2];

    const int first_child = atomic_add(node_count, 4);
    node->first_child = first_child;
    for (int q = 0; q < 4; q++) {
      quadtree_node_t* child = &nodes[first_child + q];
      child->min_x = (q & 1) ? mid_x : node->min_x;
      child->max_x = (q & 1) ? node->max_x : mid_x;
      child->min_y = (q & 2) ? node->min_y : mid_y;
      child->max_y = (q & 2) ? mid_y : node->max_y;
      child->begin = quadrant_offset[q];
      child->end = quadrant_offset[q] + quadrant_count[q];
      child->first_child = -1;
      child->depth = node->depth + 1;
    }
  }
  barrier();

  // Scatter the points into their quadrant's range of the output buffer.
  for (int i = begin + t; i < end; i += group_size_x()) {
    const float x = in_x[i];
    const float y = in_y[i];
    const int q = (x >= mid_x) + 2 * (y < mid_y);
    const unsigned int dst = atomic_add_local(&quadrant_offset[q], 1u);
    out_x[dst] = x;
    out_y[dst] = y;
  }
}
