#include <stdint.h>

__attribute__((visibility("default"), amdgpu_kernel)) void color_to_grayscale(unsigned char* img_out, unsigned char* img_in, int width, int height)
{
    int index =  __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() + __builtin_amdgcn_workitem_id_x();
    if (index < width * height) {
        unsigned char r = img_in[index * 3];
        unsigned char g = img_in[index * 3 + 1];
        unsigned char b = img_in[index * 3 + 2];
        img_out[index] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
