#include <stdint.h>

__attribute__((visibility("default"), amdgpu_kernel))
void image_blur_monochrome(
    unsigned char* img_out, const unsigned char* img_in, int width, int height) {
    
    int x = __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() +
            __builtin_amdgcn_workitem_id_x();
    int y = __builtin_amdgcn_workgroup_id_y() * __builtin_amdgcn_workgroup_size_y() +
            __builtin_amdgcn_workitem_id_y();

    if (x >= width || y >= height) return;

    int sum = 0;
    int count = 0;

    // Apply a simple 3x3 box blur kernel
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += img_in[ny * width + nx];
                count++;
            }
        }
    }

    img_out[y * width + x] = sum / count;
}

__attribute__((visibility("default"), amdgpu_kernel))
void image_blur_rgb(
    unsigned char* img_out, const unsigned char* img_in, int width, int height) {
    
    int x = __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() +
            __builtin_amdgcn_workitem_id_x();
    int y = __builtin_amdgcn_workgroup_id_y() * __builtin_amdgcn_workgroup_size_y() +
            __builtin_amdgcn_workitem_id_y();

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3; // Each pixel has 3 channels (R, G, B)

    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

    // Apply a simple 3x3 box blur kernel
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = (ny * width + nx) * 3;
                sum_r += img_in[n_idx];
                sum_g += img_in[n_idx + 1];
                sum_b += img_in[n_idx + 2];
                count++;
            }
        }
    }

    img_out[idx]     = sum_r / count;
    img_out[idx + 1] = sum_g / count;
    img_out[idx + 2] = sum_b / count;
}

