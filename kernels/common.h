#ifndef HANSA_KERNELS_COMMON_H
#define HANSA_KERNELS_COMMON_H

// Shared helpers for the kernels in this directory. Everything here is
// static inline / macros so that including it from multiple kernel files does
// not produce duplicate symbols in libkernels.so.

#define KERNEL __attribute__((visibility("default"), amdgpu_kernel)) void

// LDS ("shared memory") address space. Variables declared with SHARED are
// allocated once per workgroup. loader_uninitialized is required because C
// would otherwise zero-initialize the static, which amdgcn does not support
// for LDS.
#define LOCAL __attribute__((address_space(3)))
#define SHARED static __attribute__((address_space(3), loader_uninitialized))

// Memory scopes understood by the __hip_atomic_* builtins (only predefined
// when compiling HIP).
#ifndef __HIP_MEMORY_SCOPE_WORKGROUP
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#endif
#ifndef __HIP_MEMORY_SCOPE_AGENT
#define __HIP_MEMORY_SCOPE_AGENT 4
#endif

// Device-wide relaxed atomics (global memory), like CUDA's atomicAdd & co.
#define atomic_add(ptr, val)                             \
  __hip_atomic_fetch_add((ptr), (val), __ATOMIC_RELAXED, \
                         __HIP_MEMORY_SCOPE_AGENT)
#define atomic_cas(ptr, expected, desired)                                   \
  ({                                                                         \
    __typeof__(*(ptr)) __expected = (expected);                              \
    __hip_atomic_compare_exchange_strong((ptr), &__expected, (desired),      \
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED, \
                                         __HIP_MEMORY_SCOPE_AGENT);          \
    __expected;                                                              \
  })

// Workgroup-scoped relaxed atomics, for counters living in LDS.
#define atomic_add_local(ptr, val)                       \
  __hip_atomic_fetch_add((ptr), (val), __ATOMIC_RELAXED, \
                         __HIP_MEMORY_SCOPE_WORKGROUP)

// Equivalent of CUDA's __syncthreads(): s_barrier on its own does not order
// memory, so wrap it in workgroup-scope release/acquire fences.
static inline void
barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

static inline unsigned
local_id_x(void) {
  return __builtin_amdgcn_workitem_id_x();
}
static inline unsigned
local_id_y(void) {
  return __builtin_amdgcn_workitem_id_y();
}
static inline unsigned
local_id_z(void) {
  return __builtin_amdgcn_workitem_id_z();
}

static inline unsigned
group_id_x(void) {
  return __builtin_amdgcn_workgroup_id_x();
}
static inline unsigned
group_id_y(void) {
  return __builtin_amdgcn_workgroup_id_y();
}
static inline unsigned
group_id_z(void) {
  return __builtin_amdgcn_workgroup_id_z();
}

static inline unsigned
group_size_x(void) {
  return __builtin_amdgcn_workgroup_size_x();
}
static inline unsigned
group_size_y(void) {
  return __builtin_amdgcn_workgroup_size_y();
}
static inline unsigned
group_size_z(void) {
  return __builtin_amdgcn_workgroup_size_z();
}

static inline unsigned
global_id_x(void) {
  return group_id_x() * group_size_x() + local_id_x();
}
static inline unsigned
global_id_y(void) {
  return group_id_y() * group_size_y() + local_id_y();
}
static inline unsigned
global_id_z(void) {
  return group_id_z() * group_size_z() + local_id_z();
}

// HSA grid sizes are expressed in work-items, not workgroups.
static inline unsigned
global_size_x(void) {
  return __builtin_amdgcn_grid_size_x();
}
static inline unsigned
num_groups_x(void) {
  return (global_size_x() + group_size_x() - 1) / group_size_x();
}

static inline int
min_int(int a, int b) {
  return a < b ? a : b;
}
static inline int
max_int(int a, int b) {
  return a > b ? a : b;
}

#endif  // HANSA_KERNELS_COMMON_H
