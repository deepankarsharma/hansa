#include "common.h"

// Tiled 2D convolution. Each workgroup loads an IN_TILE_DIM x IN_TILE_DIM
// input tile (output tile plus a FILTER_RADIUS halo on every side) into LDS,
// then the inner OUT_TILE_DIM x OUT_TILE_DIM work-items each compute one
// output pixel from LDS. Halo work-items only help with loading.
//
// filter is a (2 * FILTER_RADIUS + 1)^2 row-major matrix.
// Workgroup: {IN_TILE_DIM, IN_TILE_DIM, 1}
// Grid:      {ceil(width / OUT_TILE_DIM) * IN_TILE_DIM,
//             ceil(height / OUT_TILE_DIM) * IN_TILE_DIM, 1}

#define FILTER_RADIUS 2
#define FILTER_DIM (2 * FILTER_RADIUS + 1)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

KERNEL
convolution_2d_tiled(float* out, const float* in, const float* filter,
                     int width, int height) {
  SHARED float in_tile[IN_TILE_DIM][IN_TILE_DIM];

  const int tx = local_id_x();
  const int ty = local_id_y();
  const int col = group_id_x() * OUT_TILE_DIM + tx - FILTER_RADIUS;
  const int row = group_id_y() * OUT_TILE_DIM + ty - FILTER_RADIUS;

  // Load the input tile, using 0 for ghost cells.
  if (row >= 0 && row < height && col >= 0 && col < width) {
    in_tile[ty][tx] = in[row * width + col];
  } else {
    in_tile[ty][tx] = 0.0f;
  }
  barrier();

  const int tile_col = tx - FILTER_RADIUS;
  const int tile_row = ty - FILTER_RADIUS;
  if (row >= 0 && row < height && col >= 0 && col < width && tile_row >= 0 &&
      tile_row < OUT_TILE_DIM && tile_col >= 0 && tile_col < OUT_TILE_DIM) {
    float sum = 0.0f;
    for (int f_row = 0; f_row < FILTER_DIM; f_row++) {
      for (int f_col = 0; f_col < FILTER_DIM; f_col++) {
        sum += filter[f_row * FILTER_DIM + f_col] *
               in_tile[tile_row + f_row][tile_col + f_col];
      }
    }
    out[row * width + col] = sum;
  }
}
