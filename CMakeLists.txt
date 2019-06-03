cmake_minimum_required(VERSION 3.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(main)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(default_build_type "Debug")

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/main.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

# Modern CMake
if(USE_CUDA)
    enable_language("CUDA")
    add_executable(main ${cpu_source_files} ${gpu_source_files})
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
