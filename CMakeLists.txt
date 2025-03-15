cmake_minimum_required(VERSION 3.10)
project(hansa)

# Set the project name
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(GPU_ARCH "gfx1150")

# add_compile_options("-###")
add_compile_options("-v")

# TODO: Detect architecture like gfx1103 rather than hardcoding it

find_program(CLANG_EXECUTABLE NAMES amdclang)

if(NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR "Clang not found. Please install Clang or set the CLANG_EXECUTABLE variable.")
endif()

# Use glob to find all .c files in the kernels subdirectory
file(GLOB KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.c")

# Print the found files (optional, for debugging)
message(STATUS "Kernel source files: ${KERNEL_SOURCES}")


set_source_files_properties(
    ${KERNEL_SOURCES}
    PROPERTIES COMPILE_FLAGS "-nogpulib -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=${GPU_ARCH} -O3"
)

add_library(kernels MODULE ${KERNEL_SOURCES})
set_target_properties(kernels PROPERTIES
    LINK_FLAGS "-nogpulib -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=${GPU_ARCH} -O3"
)

# Generate assembly files for kernel sources
foreach(kernel_source ${KERNEL_SOURCES})
    get_filename_component(kernel_name ${kernel_source} NAME_WE)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.s
            COMMAND ${CLANG_EXECUTABLE} -S -nogpulib -emit-llvm -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=${GPU_ARCH} -O0
            -o ${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.s ${kernel_source}
            DEPENDS ${kernel_source}
            COMMENT "Generating assembly for ${kernel_name}"
    )
    list(APPEND KERNEL_ASM_FILES ${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.s)
endforeach()


# Create .co files (as before)
foreach(kernel_source ${KERNEL_SOURCES})
    get_filename_component(kernel_name ${kernel_source} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.co
        COMMAND ${CLANG_EXECUTABLE} -c -nogpulib -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=${GPU_ARCH} -O3
        -o ${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.co ${kernel_source}
        DEPENDS ${kernel_source}
        COMMENT "Generating code object for ${kernel_name}"
    )
    list(APPEND KERNEL_CO_FILES ${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.co)
endforeach()

# Add a custom target for assembly generation
add_custom_target(kernel_asm ALL DEPENDS ${KERNEL_ASM_FILES})
add_custom_target(kernel_co ALL DEPENDS ${KERNEL_CO_FILES})

# Add the executable
add_executable(hansa main.cpp)
target_include_directories(hansa PRIVATE /opt/rocm/include)
target_link_directories(hansa PRIVATE /opt/rocm/lib)
target_link_libraries(hansa PRIVATE hsa-runtime64)
add_dependencies(hansa kernels)
add_dependencies(hansa kernel_asm)
add_dependencies(hansa kernel_co)
