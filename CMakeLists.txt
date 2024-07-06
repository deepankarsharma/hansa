cmake_minimum_required(VERSION 3.10)
project(hansa)

# Set the project name
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO: Detect architecture like gfx1103 rather than hardcoding it

find_program(CLANG_EXECUTABLE NAMES clang)

if(NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR "Clang not found. Please install Clang or set the CLANG_EXECUTABLE variable.")
endif()


add_custom_command(
        OUTPUT kernel.s
        COMMAND ${CLANG_EXECUTABLE} -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=gfx1103 -S -O3 ${CMAKE_CURRENT_SOURCE_DIR}/kernel.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.c
        COMMENT "Compiling kernel.s"
)

add_custom_command(
        OUTPUT kernel.o
        COMMAND ${CLANG_EXECUTABLE} -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=gfx1103 -c -O3 ${CMAKE_CURRENT_SOURCE_DIR}/kernel.c -o kernel.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.c
        COMMENT "Compiling kernel.c"
)

add_custom_command(
        OUTPUT kernel.co
        COMMAND ${CLANG_EXECUTABLE} -fvisibility=default -target amdgcn-amd-amdhsa -mcpu=gfx1103 -O3 kernel.o -o kernel.co
        DEPENDS kernel.o kernel.s
        COMMENT "Linking kernel.o"
)

add_custom_target(hsa_kernel DEPENDS kernel.co)

# Add the executable
add_executable(hansa main.cpp kernel.co)
target_include_directories(hansa PRIVATE /opt/rocm/include)
target_link_directories(hansa PRIVATE /opt/rocm/lib)
target_link_libraries(hansa PRIVATE hsa-runtime64)
add_dependencies(hansa hsa_kernel)