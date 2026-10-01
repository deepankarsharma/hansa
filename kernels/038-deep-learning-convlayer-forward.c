#include "common.h"

// Forward pass of a convolutional layer (no padding, stride 1).
//
//   X: input feature maps   [N][C][H][W]
//   F: filter bank          [M][C][K][K]
//   Y: output feature maps  [N][M][H_out][W_out], H_out = H - K + 1,
//                                                 W_out = W - K + 1
//
// Each workgroup computes one TILE_WIDTH x TILE_WIDTH tile of one output
// feature map of one sample:
//   group_id_x = output feature map m
//   group_id_y = tile index within the map (row-major over a
//                W_grid x H_grid grid of tiles)
//   group_id_z = sample n
//
// Workgroup: {TILE_WIDTH, TILE_WIDTH, 1}
// Grid:      {M * TILE_WIDTH, W_grid * H_grid * TILE_WIDTH, N} with
//            W_grid = ceil(W_out / TILE_WIDTH), H_grid = ceil(H_out /
//            TILE_WIDTH)

#define TILE_WIDTH 16

KERNEL
convlayer_forward(float* Y, const float* X, const float* F, int C, int H, int W,
                  int K, int M) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

  const int m = group_id_x();
  const int n = group_id_z();
  const int h = (group_id_y() / W_grid) * TILE_WIDTH + local_id_y();
  const int w = (group_id_y() % W_grid) * TILE_WIDTH + local_id_x();
  if (h >= H_out || w >= W_out) return;

  float acc = 0.0f;
  for (int c = 0; c < C; c++) {
    const float* x = X + ((long)n * C + c) * H * W;
    const float* f = F + ((long)m * C + c) * K * K;
    for (int p = 0; p < K; p++) {
      for (int q = 0; q < K; q++) {
        acc += x[(h + p) * W + (w + q)] * f[p * K + q];
      }
    }
  }
  Y[(((long)n * M + m) * H_out + h) * W_out + w] = acc;
}

// 2x2 max-pooling subsampling layer, usually run after the convolution:
//   Y: [N][M][H][W] -> P: [N][M][H / 2][W / 2]
// Workgroup: e.g. {16, 16, 1}
// Grid:      {W / 2, H / 2, N * M}
KERNEL
maxpool_2x2_forward(float* P, const float* Y, int H, int W, int num_maps) {
  const int w = global_id_x();
  const int h = global_id_y();
  const int map = global_id_z();
  const int H_p = H / 2;
  const int W_p = W / 2;
  if (w >= W_p || h >= H_p || map >= num_maps) return;

  const float* y = Y + (long)map * H * W + (2 * h) * W + 2 * w;
  P[((long)map * H_p + h) * W_p + w] = __builtin_fmaxf(
      __builtin_fmaxf(y[0], y[1]), __builtin_fmaxf(y[W], y[W + 1]));
}
