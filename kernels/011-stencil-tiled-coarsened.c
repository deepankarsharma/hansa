#include "common.h"

// Tiled 3D seven-point stencil with thread coarsening in z. A 2D workgroup
// marches through OUT_TILE_DIM z-planes, keeping only three planes (previous,
// current, next) in LDS. This allows a much larger x-y tile than 010 for the
// same amount of LDS and work-items.
//
// Workgroup: {IN_TILE_DIM, IN_TILE_DIM, 1}
// Grid:      {ceil(n / OUT_TILE_DIM) * IN_TILE_DIM,
//             ceil(n / OUT_TILE_DIM) * IN_TILE_DIM,
//             ceil(n / OUT_TILE_DIM)}

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

KERNEL
stencil_tiled_coarsened(float* out, const float* in, int n, float c0, float c1,
                        float c2, float c3, float c4, float c5, float c6) {
  SHARED float in_prev_s[IN_TILE_DIM][IN_TILE_DIM];
  SHARED float in_curr_s[IN_TILE_DIM][IN_TILE_DIM];
  SHARED float in_next_s[IN_TILE_DIM][IN_TILE_DIM];

  const int tx = local_id_x();
  const int ty = local_id_y();
  const int i_start = group_id_z() * OUT_TILE_DIM;
  const int j = group_id_y() * OUT_TILE_DIM + ty - 1;
  const int k = group_id_x() * OUT_TILE_DIM + tx - 1;
  const int jk_in_range = j >= 0 && j < n && k >= 0 && k < n;

  if (jk_in_range && i_start - 1 >= 0 && i_start - 1 < n) {
    in_prev_s[ty][tx] = in[(i_start - 1) * n * n + j * n + k];
  }
  if (jk_in_range && i_start < n) {
    in_curr_s[ty][tx] = in[i_start * n * n + j * n + k];
  }

  for (int i = i_start; i < i_start + OUT_TILE_DIM; i++) {
    if (jk_in_range && i + 1 < n) {
      in_next_s[ty][tx] = in[(i + 1) * n * n + j * n + k];
    }
    barrier();

    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
      if (ty >= 1 && ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1) {
        out[i * n * n + j * n + k] =
            c0 * in_curr_s[ty][tx] + c1 * in_curr_s[ty][tx - 1] +
            c2 * in_curr_s[ty][tx + 1] + c3 * in_curr_s[ty - 1][tx] +
            c4 * in_curr_s[ty + 1][tx] + c5 * in_prev_s[ty][tx] +
            c6 * in_next_s[ty][tx];
      }
    }
    barrier();

    // Rotate the planes. Each work-item only touches its own element, and
    // the barrier above guarantees nobody is still reading them.
    in_prev_s[ty][tx] = in_curr_s[ty][tx];
    in_curr_s[ty][tx] = in_next_s[ty][tx];
  }
}
