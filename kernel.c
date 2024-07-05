#include <stdint.h>

__attribute__((visibility("default"), amdgpu_kernel)) void meaning(float* output, float x)
{
    *output = 32 + x;
}