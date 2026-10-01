#include "common.h"

// Tiled 3D seven-point stencil. Each workgroup loads an IN_TILE_DIM^3 input
// tile (output tile plus a one-element halo) into LDS and the inner
// OUT_TILE_DIM^3 work-items compute output points from LDS.
//
// Workgroup: {IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM}
// Grid:      {ceil(n / OUT_TILE_DIM) * IN_TILE_DIM} in every dimension

#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

KERNEL
stencil_tiled(float* out, const float* in, int n, float c0, float c1, float c2,
              float c3, float c4, float c5, float c6) {
  SHARED float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

  const int tx = local_id_x();
  const int ty = local_id_y();
  const int tz = local_id_z();
  const int i = group_id_z() * OUT_TILE_DIM + tz - 1;
  const int j = group_id_y() * OUT_TILE_DIM + ty - 1;
  const int k = group_id_x() * OUT_TILE_DIM + tx - 1;

  if (i >= 0 && i < n && j >= 0 && j < n && k >= 0 && k < n) {
    in_s[tz][ty][tx] = in[i * n * n + j * n + k];
  } else {
    in_s[tz][ty][tx] = 0.0f;
  }
  barrier();

  if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
    if (tz >= 1 && tz < IN_TILE_DIM - 1 && ty >= 1 && ty < IN_TILE_DIM - 1 &&
        tx >= 1 && tx < IN_TILE_DIM - 1) {
      out[i * n * n + j * n + k] =
          c0 * in_s[tz][ty][tx] + c1 * in_s[tz][ty][tx - 1] +
          c2 * in_s[tz][ty][tx + 1] + c3 * in_s[tz][ty - 1][tx] +
          c4 * in_s[tz][ty + 1][tx] + c5 * in_s[tz - 1][ty][tx] +
          c6 * in_s[tz + 1][ty][tx];
    }
  }
}
