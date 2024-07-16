//#include <stdint.h>

#define IMPLICITARG(T) ((const T *)__builtin_amdgcn_implicitarg_ptr())

__attribute__((visibility("default"), amdgpu_kernel)) void add_arrays(int* input_a, int* input_b, int* output)
{
    //int index =  __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
    // int index = IMPLICITARG(unsigned long)[5];
    //int sz = __builtin_amdgcn_workgroup_id_x() < IMPLICITARG(unsigned short)[0] ? IMPLICITARG(unsigned short)[6] : IMPLICITARG(unsigned short)[9];
    int index =  __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
    //output[index] = input_a[index] + input_b[index];
    output[index] = IMPLICITARG(int)[0];
    //output[index] = __builtin_amdgcn_workgroup_size_x(); //__builtin_amdgcn_workgroup_size_x();
}
