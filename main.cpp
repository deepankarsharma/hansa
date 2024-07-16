#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <random>
#include <string>
#include <numeric>
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define HSA_ENFORCE(msg, rtn) \
if(rtn != HSA_STATUS_SUCCESS) {\
const char * err; \
hsa_status_string(rtn, &err); \
std::cerr<<"ERROR:"<<msg<<", rtn:"<<rtn<<", "<<err<<std::endl;\
return HSA_STATUS_ERROR; \
}

#define HSA_ENFORCE_PTR(msg, ptr) \
if(!ptr) {\
std::cerr<<"ERROR:"<<msg<<std::endl;\
return -1; \
}

class Engine;


void our_hsa_free(void *mem) {
    if (mem)
        hsa_memory_free(mem);
}

void *our_hsa_alloc(size_t size, void *param) {
    auto region = static_cast<hsa_region_t *>(param);
    void *p = nullptr;
    hsa_status_t status = hsa_memory_allocate(*region, size, (void **) &p);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "hsa_memory_allocate failed, " << status << std::endl;
        return nullptr;
    }
    return p;
}

hsa_status_t get_agent_callback(hsa_agent_t agent, void *data);

hsa_status_t get_region_callback(hsa_region_t region, void *data);

class Image {
    explicit Image(std::string path) : path_(std::move(path)), width_(0), height_(0), data_(nullptr) {
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
    unsigned char* data_;
};

class Engine {
public:
    class KernelDispatchConfig {
    public:
        KernelDispatchConfig(): grid_size{0}, workgroup_size{0}, kernel_arg_size_(0) {
        }

        KernelDispatchConfig(
            std::string code_file_name,
            std::string kernel_symbol,
            const std::array<int, 3> &grid_size,
            const std::array<int, 3> &workgroup_size,
            const int kernel_arg_size
            ) :
        code_file_name(std::move(code_file_name)),
        kernel_symbol(std::move(kernel_symbol)),
        grid_size(grid_size),
        workgroup_size(workgroup_size),
        kernel_arg_size_(kernel_arg_size)
        {
        }

        std::string code_file_name;
        std::string kernel_symbol;

        std::array<int, 3> grid_size;
        std::array<int, 3> workgroup_size;
        int kernel_arg_size_;

        [[nodiscard]] size_t size() const {
            return kernel_arg_size_;
        }
    };

public:
    friend hsa_status_t get_agent_callback(hsa_agent_t agent, void *data);

    friend hsa_status_t get_region_callback(hsa_region_t region, void *data);

    Engine() : agent_(0), cpu_agent_(0), queue_size_(0), queue_(nullptr), signal_(0), system_region_(0),
               kernarg_region_(0),
               local_region_(0), gpu_local_region_(0), aql_(nullptr), packet_index_(0), code_object_(0), executable_(0),
               group_static_size_(0) {
    }

    ~Engine() = default;

    int init() {
        hsa_status_t status = hsa_init();
        HSA_ENFORCE("hsa_init", status);

        status = hsa_iterate_agents(get_agent_callback, this);
        HSA_ENFORCE("hsa_iterate_agents", status);

        char agent_name[64];

        status = hsa_agent_get_info(agent_, HSA_AGENT_INFO_NAME, agent_name);
        HSA_ENFORCE("hsa_agent_get_info(HSA_AGENT_INFO_NAME)", status);

        std::cout << "Using agent: " << agent_name << std::endl;

        status = hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size_);
        HSA_ENFORCE("hsa_agent_get_info(HSA_AGENT_INFO_QUEUE_MAX_SIZE", status);

        status = hsa_queue_create(agent_, queue_size_, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr,UINT32_MAX, UINT32_MAX,
                                  &queue_);

        HSA_ENFORCE("hsa_queue_create", status);

        status = hsa_signal_create(1, 0, nullptr, &signal_);
        HSA_ENFORCE("hsa_signal_create", status);

        status = hsa_agent_iterate_regions(agent_, get_region_callback, this);
        HSA_ENFORCE("hsa_agent_iterate_regions", status);
        HSA_ENFORCE_PTR("Failed to find kernarg memory region", kernarg_region_.handle)

        return 0;
    }

    int setup_dispatch(const KernelDispatchConfig *cfg) {
        packet_index_ = hsa_queue_add_write_index_relaxed(queue_, 1);
        const uint32_t queue_mask = queue_->size - 1;
        aql_ = static_cast<hsa_kernel_dispatch_packet_t *>(queue_->base_address) + (packet_index_ & queue_mask);

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
        if (0 != load_bin_from_file(cfg->code_file_name.c_str()))
            return -1;
        hsa_status_t status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                                    nullptr, &executable_);
        HSA_ENFORCE("hsa_executable_create", status);
        // Load code object
        status = hsa_executable_load_code_object(executable_, agent_, code_object_, nullptr);
        HSA_ENFORCE("hsa_executable_load_code_object", status);

        // Freeze executable
        status = hsa_executable_freeze(executable_, nullptr);
        HSA_ENFORCE("hsa_executable_freeze", status);

        // Get symbol handle
        hsa_executable_symbol_t kernel_symbol;
        status = hsa_executable_get_symbol(executable_, nullptr, cfg->kernel_symbol.c_str(), agent_,
                                           0, &kernel_symbol);
        HSA_ENFORCE("hsa_executable_get_symbol", status);

        // Get code handle
        uint64_t code_handle;
        status = hsa_executable_symbol_get_info(kernel_symbol,
                                                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                &code_handle);
        HSA_ENFORCE("hsa_executable_symbol_get_info", status);
        status = hsa_executable_symbol_get_info(kernel_symbol,
                                                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                                &group_static_size_);
        HSA_ENFORCE("hsa_executable_symbol_get_info", status);

        uint32_t kernel_arg_size;
        status = hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                                &kernel_arg_size);
        HSA_ENFORCE("HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE", status);
        std::cout << "Kernel arg size: " << kernel_arg_size << std::endl;

        aql_->kernel_object = code_handle;

        // kernel args
        void *kernarg;
        status = hsa_memory_allocate(kernarg_region_, cfg->size(), &kernarg);
        HSA_ENFORCE("hsa_memory_allocate", status);
        aql_->kernarg_address = kernarg;

        std::cout << "Workgroup sizes: " << cfg->workgroup_size[0] << " " << cfg->workgroup_size[1] << " " << cfg->workgroup_size[2] << std::endl;
        aql_->workgroup_size_x = cfg->workgroup_size[0];
        aql_->workgroup_size_y = cfg->workgroup_size[1];
        aql_->workgroup_size_z = cfg->workgroup_size[2];

        std::cout << "Grid sizes: " << cfg->grid_size[0] << " " << cfg->grid_size[1] << " " << cfg->grid_size[2] << std::endl;

        aql_->grid_size_x = cfg->grid_size[0];
        aql_->grid_size_y = cfg->grid_size[1];
        aql_->grid_size_z = cfg->grid_size[2];
        return status;
    }

    void *kernarg_address() {
        return aql_->kernarg_address;
    }

    void *alloc_local(int size) {
        return our_hsa_alloc(size, &this->local_region_);
    }

    int load_bin_from_file(const char *file_name) {
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
                  std::istreambuf_iterator<char>(),
                  ptr);

        const hsa_status_t status = hsa_code_object_deserialize(ptr, size, nullptr, &code_object_);
        HSA_ENFORCE("hsa_code_object_deserialize", status);

        return 0;
    }

    int dispatch() {
        uint16_t header =
                (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                (1 << HSA_PACKET_HEADER_BARRIER) |
                (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

        // total dimension
        uint16_t dim = 1;
        if (aql_->grid_size_y > 1)
            dim = 2;
        if (aql_->grid_size_z > 1)
            dim = 3;
        aql_->group_segment_size = group_static_size_;
        const uint16_t setup = dim << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        const uint32_t header32 = header | (setup << 16);

        __atomic_store_n(reinterpret_cast<uint32_t *>(aql_), header32, __ATOMIC_RELEASE);
        hsa_signal_store_relaxed(queue_->doorbell_signal, static_cast<hsa_signal_value_t>(packet_index_));

        return 0;
    }

    hsa_signal_value_t wait() {
        //return hsa_signal_wait_acquire(signal_, HSA_SIGNAL_CONDITION_EQ, 0, ~0ULL, HSA_WAIT_STATE_ACTIVE);
        return hsa_signal_wait_acquire(signal_, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
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


hsa_status_t get_agent_callback(const hsa_agent_t agent, void *data) {
    if (!data)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    hsa_device_type_t hsa_device_type;
    hsa_status_t hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
    if (hsa_error_code != HSA_STATUS_SUCCESS)
        return hsa_error_code;

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

hsa_status_t get_region_callback(const hsa_region_t region, void *data) {
    hsa_region_segment_t segment_id;
    hsa_status_t status = hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);
    HSA_ENFORCE("Failed getting region info", status);

    if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags;
    bool host_accessible_region = false;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    hsa_region_get_info(region, static_cast<hsa_region_info_t>(HSA_AMD_REGION_INFO_HOST_ACCESSIBLE),
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


int kernel() {
    Engine engine;

    long rtn = engine.init();
    if (rtn) {
        std::cout << "Failed to initialize engine" << std::endl;
        return -1;
    }

    constexpr int num_elements = 100000;
    std::vector<int> input_a(num_elements);
    std::vector<int> input_b(num_elements);

    std::iota(input_a.begin(), input_a.end(), 0);
    std::iota(input_b.begin(), input_b.end(), 0);

    std::cout << "Engine init: OK" << std::endl;

    struct alignas(16) args_t {
        int *input_a;
        int *input_b;
        int *output;
    };

    auto device_input_a = (int *) engine.alloc_local(num_elements * sizeof(int));
    auto device_input_b = (int *) engine.alloc_local(num_elements * sizeof(int));
    auto device_output = (int *) engine.alloc_local(num_elements * sizeof(int));

    memcpy(device_input_a, input_a.data(), num_elements * sizeof(int));
    memcpy(device_input_b, input_b.data(), num_elements * sizeof(int));

    args_t args{.input_a = device_input_a, .input_b = device_input_b, .output = device_output};

    Engine::KernelDispatchConfig d_param(
        "libkernels.so", // kernel compiled object name,
        "add_arrays.kd", // name of kernel
        {num_elements, 1, 1}, // grid size
        {64, 1, 1}, // workgroup size
        sizeof(args_t)
    );

    rtn = engine.setup_dispatch(&d_param);
    if (rtn) return -1;
    std::cout << "Setup dispatch: OK" << std::endl;

    memcpy(engine.kernarg_address(), &args, sizeof(args));

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
            << (num_elements - 1) * num_elements
            << ". Calculated sum is "
            << std::reduce(device_output, device_output + num_elements, 0)
            << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    // Launch a kernel on our AMD GPU/APU
    return kernel();
}
