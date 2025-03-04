#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#define HSA_ENFORCE(msg, rtn)                                      \
  if (rtn != HSA_STATUS_SUCCESS) {                                 \
    const char *err;                                               \
    hsa_status_string(rtn, &err);                                  \
    std::cerr << "ERROR:" << msg << ", rtn:" << rtn << ", " << err \
              << std::endl;                                        \
    return HSA_STATUS_ERROR;                                       \
  }

#define HSA_ENFORCE_PTR(msg, ptr)              \
  if (!ptr) {                                  \
    std::cerr << "ERROR:" << msg << std::endl; \
    return -1;                                 \
  }

class Engine;

#pragma pack(push, 1)
typedef struct ImplicitArg_s {
  uint32_t block_count_x;
  uint32_t block_count_y;
  uint32_t block_count_z;

  uint16_t group_size_x;
  uint16_t group_size_y;
  uint16_t group_size_z;

  uint16_t remainder_x;
  uint16_t remainder_y;
  uint16_t remainder_z;

  uint64_t tool_correlation_id;
  uint64_t reserved_1;

  uint64_t global_offset_x;
  uint64_t global_offset_y;
  uint64_t global_offset_z;

  uint16_t grid_dims;

  uint16_t reserved_2;
  uint16_t reserved_3;
  uint16_t reserved_4;

  uint64_t printf_buffer;
  uint64_t hostcall_buffer;
  uint64_t multigrid_sync_arg;
  uint64_t heap_v1;
  uint64_t default_queue;
  uint64_t completion_action;
  uint32_t dynamic_lds_size;
  uint32_t private_base;
  uint32_t shared_base;
} ImplicitArg;
#pragma pack(pop)

void
our_hsa_free(void *mem) {
  if (mem) hsa_memory_free(mem);
}

void *
our_hsa_alloc(size_t size, void *param) {
  auto region = static_cast<hsa_region_t *>(param);
  void *p = nullptr;
  hsa_status_t status = hsa_memory_allocate(*region, size, (void **)&p);
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "hsa_memory_allocate failed, " << status << std::endl;
    return nullptr;
  }
  return p;
}

hsa_status_t
get_agent_callback(hsa_agent_t agent, void *data);

hsa_status_t
get_region_callback(hsa_region_t region, void *data);

class Image {
  explicit Image(std::string path)
      : path_(std::move(path)), width_(0), height_(0), data_(nullptr) {
    data_ = stbi_load(path_.c_str(), &width_, &height_, nullptr, 0);
    if (!data_) {
      std::cerr << "Failed to load image: " << path_ << std::endl;
    }
  }

  ~Image() {
    if (data_) {
      stbi_image_free(data_);
    }
  }

  std::string path_;
  int width_;
  int height_;
  unsigned char *data_;
};

class Engine {
 public:
  class KernelDispatchConfig {
   public:
    KernelDispatchConfig()
        : grid_size{0}, workgroup_size{0}, kernel_arg_size_(0) {}

    KernelDispatchConfig(std::string code_file_name, std::string kernel_symbol,
                         const std::array<int, 3> &grid_size,
                         const std::array<int, 3> &workgroup_size,
                         const int kernel_arg_size)
        : code_file_name(std::move(code_file_name)),
          kernel_symbol(std::move(kernel_symbol)),
          grid_size(grid_size),
          workgroup_size(workgroup_size),
          kernel_arg_size_(kernel_arg_size) {}

    std::string code_file_name;
    std::string kernel_symbol;

    std::array<int, 3> grid_size;
    std::array<int, 3> workgroup_size;
    int kernel_arg_size_;

    [[nodiscard]]
    size_t
    size() const {
      return kernel_arg_size_;
    }
  };

 public:
  friend hsa_status_t
  get_agent_callback(hsa_agent_t agent, void *data);

  friend hsa_status_t
  get_region_callback(hsa_region_t region, void *data);

  Engine()
      : agent_(0),
        cpu_agent_(0),
        queue_size_(0),
        queue_(nullptr),
        signal_(0),
        system_region_(0),
        kernarg_region_(0),
        local_region_(0),
        gpu_local_region_(0),
        aql_(nullptr),
        packet_index_(0),
        code_object_(0),
        executable_(0),
        group_static_size_(0) {}

  ~Engine() = default;

  int
  init() {
    hsa_status_t status = hsa_init();
    HSA_ENFORCE("hsa_init", status);

    status = hsa_iterate_agents(get_agent_callback, this);
    HSA_ENFORCE("hsa_iterate_agents", status);

    char agent_name[64];

    status = hsa_agent_get_info(agent_, HSA_AGENT_INFO_NAME, agent_name);
    HSA_ENFORCE("hsa_agent_get_info(HSA_AGENT_INFO_NAME)", status);

    std::cout << "Using agent: " << agent_name << std::endl;

    status =
        hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size_);
    HSA_ENFORCE("hsa_agent_get_info(HSA_AGENT_INFO_QUEUE_MAX_SIZE", status);

    status =
        hsa_queue_create(agent_, queue_size_, HSA_QUEUE_TYPE_MULTI, nullptr,
                         nullptr, UINT32_MAX, UINT32_MAX, &queue_);

    HSA_ENFORCE("hsa_queue_create", status);

    status = hsa_signal_create(1, 0, nullptr, &signal_);
    HSA_ENFORCE("hsa_signal_create", status);

    status = hsa_agent_iterate_regions(agent_, get_region_callback, this);
    HSA_ENFORCE("hsa_agent_iterate_regions", status);
    HSA_ENFORCE_PTR("Failed to find kernarg memory region",
                    kernarg_region_.handle)

    return 0;
  }

  template <typename ARGS_T>
  int
  setup_dispatch(const KernelDispatchConfig *cfg, const ARGS_T &args) {
    packet_index_ = hsa_queue_add_write_index_relaxed(queue_, 1);
    const uint32_t queue_mask = queue_->size - 1;
    aql_ = static_cast<hsa_kernel_dispatch_packet_t *>(queue_->base_address) +
           (packet_index_ & queue_mask);

    constexpr size_t aql_header_size = 4;
    memset(aql_ + aql_header_size, 0, sizeof(*aql_) - aql_header_size);

    // initialize_packet
    aql_->completion_signal = signal_;
    aql_->workgroup_size_x = 1;
    aql_->workgroup_size_y = 1;
    aql_->workgroup_size_z = 1;
    aql_->grid_size_x = 1;
    aql_->grid_size_y = 1;
    aql_->grid_size_z = 1;
    aql_->group_segment_size = 0;
    aql_->private_segment_size = 0;

    // executable
    if (0 != load_bin_from_file(cfg->code_file_name.c_str())) return -1;
    hsa_status_t status = hsa_executable_create(
        HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, nullptr, &executable_);
    HSA_ENFORCE("hsa_executable_create", status);
    // Load code object
    status = hsa_executable_load_code_object(executable_, agent_, code_object_,
                                             nullptr);
    HSA_ENFORCE("hsa_executable_load_code_object", status);

    // Freeze executable
    status = hsa_executable_freeze(executable_, nullptr);
    HSA_ENFORCE("hsa_executable_freeze", status);

    // Get symbol handle
    hsa_executable_symbol_t kernel_symbol;
    status = hsa_executable_get_symbol(executable_, nullptr,
                                       cfg->kernel_symbol.c_str(), agent_, 0,
                                       &kernel_symbol);
    HSA_ENFORCE("hsa_executable_get_symbol", status);

    // Get code handle
    uint64_t code_handle;
    status = hsa_executable_symbol_get_info(
        kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_handle);
    HSA_ENFORCE("hsa_executable_symbol_get_info", status);
    status = hsa_executable_symbol_get_info(
        kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &group_static_size_);
    HSA_ENFORCE("hsa_executable_symbol_get_info", status);

    uint32_t kernel_arg_size;
    status = hsa_executable_symbol_get_info(
        kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernel_arg_size);
    HSA_ENFORCE("HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE",
                status);
    std::cout << "Kernel arg size: " << kernel_arg_size << std::endl;

    aql_->kernel_object = code_handle;

    // kernel args
    void *kernarg;
    status = hsa_memory_allocate(kernarg_region_, kernel_arg_size, &kernarg);
    HSA_ENFORCE("hsa_memory_allocate", status);

    std::memset(kernarg, 0, kernel_arg_size);
    std::memcpy(kernarg, &args, sizeof(ARGS_T));

    bool dims = 1 + (cfg->grid_size[1] * cfg->workgroup_size[1] != 1) +
                (cfg->grid_size[2] * cfg->workgroup_size[2] != 1);
    auto implicit_args = reinterpret_cast<ImplicitArg *>(
        reinterpret_cast<std::uint8_t *>(kernarg) + sizeof(ARGS_T));

    implicit_args->block_count_x = cfg->grid_size[0];
    implicit_args->block_count_y = cfg->grid_size[1];
    implicit_args->block_count_z = cfg->grid_size[2];

    implicit_args->group_size_x = cfg->workgroup_size[0];
    implicit_args->group_size_y = cfg->workgroup_size[1];
    implicit_args->group_size_z = cfg->workgroup_size[2];

    implicit_args->grid_dims = dims;

    aql_->kernarg_address = kernarg;

    std::cout << "Workgroup sizes: " << cfg->workgroup_size[0] << " "
              << cfg->workgroup_size[1] << " " << cfg->workgroup_size[2]
              << std::endl;
    aql_->workgroup_size_x = cfg->workgroup_size[0];
    aql_->workgroup_size_y = cfg->workgroup_size[1];
    aql_->workgroup_size_z = cfg->workgroup_size[2];

    std::cout << "Grid sizes: " << cfg->grid_size[0] << " " << cfg->grid_size[1]
              << " " << cfg->grid_size[2] << std::endl;

    aql_->grid_size_x = cfg->grid_size[0];
    aql_->grid_size_y = cfg->grid_size[1];
    aql_->grid_size_z = cfg->grid_size[2];
    return status;
  }

  void *
  kernarg_address() {
    return aql_->kernarg_address;
  }

  void *
  alloc_local(int size) {
    return our_hsa_alloc(size, &this->local_region_);
  }

  int
  load_bin_from_file(const char *file_name) {
    std::ifstream inf(file_name, std::ios::binary | std::ios::ate);
    if (!inf) {
      std::cerr << "Error: failed to load " << file_name << std::endl;
      return -1;
    }
    const size_t size = static_cast<std::string::size_type>(inf.tellg());
    char *ptr = static_cast<char *>(our_hsa_alloc(size, &this->system_region_));
    HSA_ENFORCE_PTR("failed to allocate memory for code object", ptr);

    inf.seekg(0, std::ios::beg);
    std::copy(std::istreambuf_iterator<char>(inf),
              std::istreambuf_iterator<char>(), ptr);

    const hsa_status_t status =
        hsa_code_object_deserialize(ptr, size, nullptr, &code_object_);
    HSA_ENFORCE("hsa_code_object_deserialize", status);

    return 0;
  }

  int
  dispatch() {
    uint16_t header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    // total dimension
    uint16_t dim = 1;
    if (aql_->grid_size_y > 1) dim = 2;
    if (aql_->grid_size_z > 1) dim = 3;
    aql_->group_segment_size = group_static_size_;
    const uint16_t setup = dim << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    const uint32_t header32 = header | (setup << 16);

    __atomic_store_n(reinterpret_cast<uint32_t *>(aql_), header32,
                     __ATOMIC_RELEASE);
    hsa_signal_store_relaxed(queue_->doorbell_signal,
                             static_cast<hsa_signal_value_t>(packet_index_));

    return 0;
  }

  hsa_signal_value_t
  wait() {
    // return hsa_signal_wait_acquire(signal_, HSA_SIGNAL_CONDITION_EQ, 0,
    // ~0ULL, HSA_WAIT_STATE_ACTIVE);
    return hsa_signal_wait_acquire(signal_, HSA_SIGNAL_CONDITION_LT, 1,
                                   UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  }

 private:
  hsa_agent_t agent_;
  hsa_agent_t cpu_agent_;
  uint32_t queue_size_;
  hsa_queue_t *queue_;
  hsa_signal_t signal_;

  hsa_region_t system_region_;
  hsa_region_t kernarg_region_;
  hsa_region_t local_region_;
  hsa_region_t gpu_local_region_;

  hsa_kernel_dispatch_packet_t *aql_;
  uint64_t packet_index_;

  hsa_code_object_t code_object_;
  hsa_executable_t executable_;
  uint32_t group_static_size_;
};

hsa_status_t
get_agent_callback(const hsa_agent_t agent, void *data) {
  if (!data) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
  if (hsa_error_code != HSA_STATUS_SUCCESS) return hsa_error_code;

  if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {
    auto b = static_cast<Engine *>(data);
    b->agent_ = agent;
  }
  if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
    auto b = static_cast<Engine *>(data);
    b->cpu_agent_ = agent;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_region_callback(const hsa_region_t region, void *data) {
  hsa_region_segment_t segment_id;
  hsa_status_t status =
      hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);
  HSA_ENFORCE("Failed getting region info", status);

  if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  bool host_accessible_region = false;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  hsa_region_get_info(
      region,
      static_cast<hsa_region_info_t>(HSA_AMD_REGION_INFO_HOST_ACCESSIBLE),
      &host_accessible_region);

  auto b = static_cast<Engine *>(data);

  if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
    b->system_region_ = region;
  }

  if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
    if (host_accessible_region) {
      b->local_region_ = region;
    } else {
      b->gpu_local_region_ = region;
    }
  }

  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    b->kernarg_region_ = region;
  }

  return HSA_STATUS_SUCCESS;
}

int
kernel_001_vector_add() {
  Engine engine;

  long rtn = engine.init();
  if (rtn) {
    std::cout << "Failed to initialize engine" << std::endl;
    return -1;
  }

  constexpr int num_elements = 1000;
  std::vector<int> input_a(num_elements);
  std::vector<int> input_b(num_elements);

  std::iota(input_a.begin(), input_a.end(), 0);
  std::iota(input_b.begin(), input_b.end(), 0);

  std::cout << "Engine init: OK" << std::endl;

  struct args_t {
    int *input_a;
    int *input_b;
    int *output;
  };

  auto device_input_a = (int *)engine.alloc_local(num_elements * sizeof(int));
  auto device_input_b = (int *)engine.alloc_local(num_elements * sizeof(int));
  auto device_output = (int *)engine.alloc_local(num_elements * sizeof(int));

  memcpy(device_input_a, input_a.data(), num_elements * sizeof(int));
  memcpy(device_input_b, input_b.data(), num_elements * sizeof(int));

  args_t args{.input_a = device_input_a,
              .input_b = device_input_b,
              .output = device_output};

  Engine::KernelDispatchConfig d_param(
      "libkernels.so",       // kernel compiled object name,
      "add_arrays.kd",       // name of kernel
      {num_elements, 1, 1},  // grid size
      {64, 1, 1},            // workgroup size
      sizeof(args_t));

  rtn = engine.setup_dispatch(&d_param, args);
  if (rtn) return -1;
  std::cout << "Setup dispatch: OK" << std::endl;

  engine.dispatch();
  std::cout << "Dispatch: OK" << std::endl;

  rtn = engine.wait();
  if (rtn) return -1;
  std::cout << "Wait: OK" << std::endl;

  int sum = 0;
  for (int i = 0; i < num_elements; ++i) {
    sum += device_output[i];
  }
  // Sum of numbers 0..n is n * (n - 1) / 2
  // In our case n = 99 - sum would be 99 * 100 / 2 = 4950
  // Since we have two arrays each that sum up to 4950,
  // we expect the sum of those two arrays to be = 2 * 4950 = 9900
  std::cout << "We expected the sum to be :"
            << (num_elements - 1) * num_elements << ". Calculated sum is "
            << std::reduce(device_output, device_output + num_elements, 0)
            << std::endl;
  return 0;
}

/// Kernel launcher for color-to-grayscale conversion
int
kernel_002_color_to_grayscale() {
  // Load the input image using stb_image.
  int width, height, channels;
  unsigned char *host_img =
      stbi_load("../data/images/teapot.jpg", &width, &height, &channels, 0);
  if (!host_img) {
    std::cout << "Failed to load image teapot.png" << std::endl;
    return -1;
  }
  if (channels < 3) {
    std::cout << "Image does not have enough channels (expected at least 3)"
              << std::endl;
    stbi_image_free(host_img);
    return -1;
  }
  std::cout << "Loaded image teapot.png: " << width << " x " << height
            << ", channels: " << channels << std::endl;

  // Create a host buffer for the grayscale output (1 channel per pixel).
  std::vector<unsigned char> host_out(width * height);

  // Initialize the Engine.
  Engine engine;
  long rtn = engine.init();
  if (rtn) {
    std::cout << "Failed to initialize engine" << std::endl;
    stbi_image_free(host_img);
    return -1;
  }
  std::cout << "Engine init: OK" << std::endl;

  // Allocate device memory.
  // Input image is color (3 channels) so total size = width * height * 3.
  auto device_input = (unsigned char *)engine.alloc_local(
      width * height * 3 * sizeof(unsigned char));
  // Output image is grayscale so total size = width * height.
  auto device_output = (unsigned char *)engine.alloc_local(
      width * height * sizeof(unsigned char));

  // Copy the host input image to the device.
  memcpy(device_input, host_img, width * height * 3 * sizeof(unsigned char));
  // Free the host image as it's now on the device.
  stbi_image_free(host_img);

  // Define the arguments structure matching the kernel signature.
  struct args_t {
    unsigned char *img_out;
    unsigned char *img_in;
    int width;
    int height;
  };

  args_t args{.img_out = device_output,
              .img_in = device_input,
              .width = width,
              .height = height};

  // Total number of pixels (each pixel gets processed by one workitem).
  int num_pixels = width * height;

  // Configure the kernel dispatch.
  Engine::KernelDispatchConfig d_param(
      "libkernels.so",          // Kernel compiled object.
      "color_to_grayscale.kd",  // Kernel name.
      {num_pixels, 1, 1},       // Grid size.
      {64, 1, 1},               // Workgroup size.
      sizeof(args_t));

  rtn = engine.setup_dispatch(&d_param, args);
  if (rtn) return -1;
  std::cout << "Setup dispatch: OK" << std::endl;

  // Launch the kernel.
  engine.dispatch();
  std::cout << "Dispatch: OK" << std::endl;

  rtn = engine.wait();
  if (rtn) return -1;
  std::cout << "Wait: OK" << std::endl;

  // Copy the grayscale output from device back to host memory.
  memcpy(host_out.data(), device_output,
         width * height * sizeof(unsigned char));

  if (stbi_write_png("teapot_grayscale.png", width, height, 1, host_out.data(),
                     width)) {
    std::cout << "Grayscale image saved as teapot_grayscale.png" << std::endl;
  } else {
    std::cout << "Failed to save grayscale image" << std::endl;
  }

  return 0;
}

int
main(int argc, char **argv) {
  kernel_001_vector_add();
  kernel_002_color_to_grayscale();
  return 0;
}
