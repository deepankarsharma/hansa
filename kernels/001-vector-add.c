//#include <stdint.h>

#define IMPLICITARG(T) ((const T *)__builtin_amdgcn_implicitarg_ptr())

__attribute__((visibility("default"), amdgpu_kernel)) void add_arrays(int* input_a, int* input_b, int* output)
{
    int index =  __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() + __builtin_amdgcn_workitem_id_x();
    output[index] = input_a[index] + input_b[index];
}
