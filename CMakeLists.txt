# Requirements
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main Project
project(proj VERSION 1.0)
FILE(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "./src/*.cpp")
add_executable(proj ${CPP_SOURCES})
target_include_directories(proj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
target_link_libraries (proj ${OpenCL_LIBRARY})
target_compile_definitions(proj PUBLIC CL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(proj PUBLIC CL_HPP_TARGET_OPENCL_VERSION=210)