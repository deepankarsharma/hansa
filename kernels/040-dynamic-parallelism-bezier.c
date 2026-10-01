#include "common.h"

// Tessellation of quadratic Bezier curves, where the number of points per
// curve depends on its curvature.
//
// The CUDA version of this example uses dynamic parallelism: a parent kernel
// computes how many points each curve needs and launches a child grid sized
// for that curve. Plain HSA has no device-side enqueue from C, so the same
// split is expressed as two kernels launched back to back by the host:
//
//   1. bezier_compute_num_vertices - one work-item per curve ("parent").
//        Grid: {num_lines, 1, 1}, workgroup e.g. {64, 1, 1}
//   2. bezier_tessellate - one workgroup per curve ("child"); its
//        work-items stride over that curve's points, so the amount of work
//        per curve adapts to its curvature.
//        Grid: {num_lines * 32, 1, 1}, workgroup {32, 1, 1}
//
// The host side bezier_line_t must match this layout (plain floats/ints, no
// padding).

#define MAX_TESS_POINTS 32

typedef struct {
  float x;
  float y;
} float2_t;

typedef struct {
  float2_t cp[3];  // Control points.
  float2_t vertex_pos[MAX_TESS_POINTS];
  int num_vertices;
} bezier_line_t;

static float
distance_to_line(float2_t p, float2_t a, float2_t b) {
  const float dx = b.x - a.x;
  const float dy = b.y - a.y;
  const float len = __builtin_sqrtf(dx * dx + dy * dy);
  if (len == 0.0f) {
    const float px = p.x - a.x;
    const float py = p.y - a.y;
    return __builtin_sqrtf(px * px + py * py);
  }
  return __builtin_fabsf(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / len;
}

KERNEL
bezier_compute_num_vertices(bezier_line_t* lines, int num_lines) {
  const int line = global_id_x();
  if (line >= num_lines) return;

  // How far the middle control point is from the chord is a cheap measure of
  // curvature.
  const float curvature =
      distance_to_line(lines[line].cp[1], lines[line].cp[0], lines[line].cp[2]);
  int num_vertices = (int)(curvature * 16.0f);
  num_vertices = min_int(max_int(num_vertices, 4), MAX_TESS_POINTS);
  lines[line].num_vertices = num_vertices;
}

KERNEL
bezier_tessellate(bezier_line_t* lines, int num_lines) {
  const int line = group_id_x();
  if (line >= num_lines) return;

  const float2_t p0 = lines[line].cp[0];
  const float2_t p1 = lines[line].cp[1];
  const float2_t p2 = lines[line].cp[2];
  const int num_vertices = lines[line].num_vertices;

  for (int i = local_id_x(); i < num_vertices; i += group_size_x()) {
    const float u = (float)i / (float)(num_vertices - 1);
    const float omu = 1.0f - u;
    const float b0 = omu * omu;
    const float b1 = 2.0f * u * omu;
    const float b2 = u * u;
    lines[line].vertex_pos[i].x = b0 * p0.x + b1 * p1.x + b2 * p2.x;
    lines[line].vertex_pos[i].y = b0 * p0.y + b1 * p1.y + b2 * p2.y;
  }
}
